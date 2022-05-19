
from lib2to3.pgen2.token import NT_OFFSET
import os, sys
from importlib_metadata import requires
from matplotlib import use

from NDNT.training.trainer import Trainer
sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis
import pickle
import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn
from scipy.io import loadmat

from datasets import GenericDataset
from NDNT.training import LBFGSTrainer

from torch.utils.data import Subset, DataLoader

from models import SharedGain, SharedLatentGain

'''
Model fitting procedure for the shared gain / offset model

'''

def get_dataloaders(ds, folds=5, batch_size=64, use_dropout=False, seed=1234):
    
    np.random.seed(seed)

    NT = len(ds)

    if use_dropout: # training and test set are the same, but differ in their datafilters
        from copy import deepcopy
        train_ds = deepcopy(ds)
        val_ds = deepcopy(ds)
        NT, NC = ds.covariates['robs'].shape
        p_holdout = 1/folds

        Mask = np.random.rand(NT, NC) > p_holdout
        train_ds.covariates['dfs'] = torch.tensor(np.logical_and(train_ds.covariates['dfs'].cpu().numpy(), Mask), dtype=torch.float32, device=ds.device)
        val_ds.covariates['dfs'] = torch.tensor(np.logical_and(val_ds.covariates['dfs'].cpu().numpy(), ~Mask), dtype=torch.float32, device=ds.device)
        train_inds = Mask
        val_inds = ~Mask
    
    else: # conventional training and test set
        n_val = NT//folds
        val_inds = np.random.choice(range(NT), size=n_val, replace=False)
        train_inds = np.setdiff1d(range(NT), val_inds)

        train_ds = Subset(ds, train_inds.tolist())
        val_ds = Subset(ds, val_inds.tolist())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, (train_inds, val_inds)
        
def rsquared(y, yhat, dfs=None):
    if dfs is None:
        dfs = torch.ones(y.shape, device=y.device)
    ybar = (y * dfs).sum(dim=0) / dfs.sum(dim=0)
    resids = y - yhat
    residnull = y - ybar
    sstot = torch.sum( residnull**2*dfs, dim=0)
    ssres = torch.sum( resids**2*dfs, dim=0)
    r2 = 1 - ssres/sstot

    return r2.detach().cpu()

def censored_lstsq(A, B, M):
    """Solves least squares problem with missing data in B
    Note: uses a broadcasted solve for speed.
    Args
    ----
    A (ndarray) : m x r matrix
    B (ndarray) : m x n matrix
    M (ndarray) : m x n binary matrix (zeros indicate missing values)
    Returns
    -------
    X (ndarray) : r x n matrix that minimizes norm(M*(AX - B))

    based off code from: http://alexhwilliams.info/itsneuronalblog/2018/02/26/crossval/
    """

    if A.ndim == 1:
        A = A[:,None]

    # else solve via tensor representation
    rhs = (A.T@(M * B)).T[:,:,None] # n x r x 1 tensor
    T = torch.matmul(A.T[None,:,:], M.T[:,:,None] * A[None,:,:]) # n x r x r tensor
    try:
        # transpose to get r x n
        return torch.squeeze(torch.linalg.solve(T, rhs), dim=-1).T
    except:
        r = T.shape[1]
        T[:,torch.arange(r),torch.arange(r)] += 1e-6
        return torch.squeeze(torch.linalg.solve(T, rhs), dim=-1).T


def pca_train(data, rank, Mtrain, max_iter):
    
    # choose solver for alternating minimization
    solver = censored_lstsq

    # initialize U randomly
    U = torch.randn(data.shape[0], rank, device=data.device)

    # return U, data, rank, Mtrain
    Vt = solver(U, data, Mtrain)
    resid = U@Vt - data
    mse0 = torch.mean(resid**2)
    tol = 1e-3
    # fit pca/nmf
    for itr in range(max_iter):
        Vt = solver(U, data, Mtrain)
        U = solver(Vt.T, data.T, Mtrain.T).T
        resid = U@Vt - data
        mse = torch.mean(resid[Mtrain]**2)
        # print('%d) %.3f' %(itr, mse))
        if mse > (mse0 - tol):
            break
        mse0 = mse

    return mse, U, Vt

def cv_pca(data, rank, Mtrain=None, Mtest=None, p_holdout=0.2, max_iter=10, replicates=5):
    """Fit PCA while holding out a fraction of the dataset.
    """

    # create masking matrix
    if Mtrain is None:
        Mtrain = torch.rand(*data.shape, device=data.device) > p_holdout
    
    if Mtest is None:
        Mtest = ~Mtrain

    Mtrain = Mtrain.to(data.device)
    Mtest = Mtest.to(data.device)

    mses = []
    Us = []
    Vts = []

    for r in range(replicates):
        mse, U, Vt = pca_train(data, rank, Mtrain, max_iter)
        mses.append(mse.item())
        Us.append(U)
        Vts.append(Vt)
    
    id = np.argmin(np.asarray(mses))
    U = Us[id]
    Vt = Vts[id]

    # return result and test/train error
    resid = U@Vt - data
    total_err = data - torch.mean(data, dim=0)
    train_err = 1 - torch.sum(resid[Mtrain]**2) / torch.sum(total_err[Mtrain]**2)
    test_err = 1 - torch.sum(resid[Mtest]**2) / torch.sum(total_err[Mtest]**2)
    return U, Vt, train_err, test_err

def fit_model(model, train_dl, val_dl,
    lr=1e-3, max_epochs=5,
    wd=0.01,
    max_iter=10000,
    use_lbfgs=False,
    verbose=0,
    early_stopping_patience=10,
    use_warmup=True,
    seed=1234,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from NDNT.training import Trainer, EarlyStopping, LBFGSTrainer
    model.prepare_regularization()

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(model.parameters(),
                    history_size=10,
                    max_iter=max_iter,
                    tolerance_change=1e-9,
                    line_search_fn=None,
                    tolerance_grad=1e-5)

        trainer = LBFGSTrainer(
            optimizer=optimizer,
            device=device,
            accumulate_grad_batches=len(train_dl),
            max_epochs=1,
            optimize_graph=True,
            log_activations=False,
            set_grad_to_none=False,
            verbose=verbose)

        trainer.fit(model, train_dl.dataset[:], seed=seed)

    else:
        earlystopping = EarlyStopping(patience=early_stopping_patience, verbose=False)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
        if use_warmup:
            optimizer.param_groups[0]['lr'] = 1e-5
            warmup_epochs = 5
            trainer = Trainer(model, optimizer,
                device=device,
                optimize_graph=True,
                early_stopping=earlystopping,
                max_epochs=warmup_epochs,
                log_activations=True,
                verbose=verbose)

            trainer.fit(model, train_dl, val_dl, seed=seed)
            trainer.optimizer.param_groups[0]['lr'] = lr
            trainer.max_epochs = max_epochs
            
            trainer.fit(model, train_dl, val_dl, seed=seed)
        else:
        # scheduler = OneCycleLR(optimizer, max_lr=lr,
        #     steps_per_epoch=len(train_dl), epochs=max_epochs)

            earlystopping = EarlyStopping(patience=early_stopping_patience, verbose=False)
            trainer = Trainer(model, optimizer,
                    device=device,
                    optimize_graph=True,
                    max_epochs=max_epochs,
                    early_stopping=earlystopping,
                    log_activations=True,
                    scheduler_after='batch',
                    scheduler_metric=None,
                    verbose=verbose)

            trainer.fit(model, train_dl, val_dl, seed=seed)
    
    return trainer

def get_data(fpath, fname, num_tents=10,
        normalize_robs=True,
        zthresh=10,
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
    from datasets.utils import tent_basis_generate

    dat = loadmat(os.path.join(fpath, fname))

    trial_ix = np.where(~np.isnan(dat['gdirection']))[0]

    directions = np.unique(dat['gdirection'][trial_ix])
    freqs = np.unique(dat['gfreq'][trial_ix])
    direction = dat['gdirection'][trial_ix, ...]
    freq = dat['gfreq'][trial_ix, ...]

    dironehot = direction==directions
    freqonehot = freq==freqs

    ndir = dironehot.shape[1]
    nfreq = freqonehot.shape[1]

    nstim = ndir*nfreq

    # stimulus is a vector of nDirections x nFreqs
    stim = np.reshape(freqonehot[...,None] * dironehot[:,None,...], [-1, nstim])
    # stim = np.reshape(dironehot[...,None] * freqonehot[:,None,...], [-1, nstim])
    # stim = np.reshape( np.expand_dims(dironehot, -1)*np.expand_dims(freqonehot, 1), [-1, nstim])

    NT = len(freq)
    xs = np.linspace(0, NT-1, num_tents)
    tents = tent_basis_generate(xs)

    robs = dat['robs'][trial_ix,:].astype(np.float32)
    s = np.std(robs, axis=0)
    mu = np.mean(robs, axis=0)
    from scipy.ndimage import uniform_filter
    robs_smooth = uniform_filter(robs, size=(10, 1), mode='constant')
    x = robs / robs_smooth

    dfs = x < zthresh
    # dfs = np.abs( (robs - mu) / s) < zthresh # filter out outliers
    if normalize_robs:
        robs = ( (robs-mu) / s)

    data = {'runningspeed': torch.tensor(dat['runningspeed'][trial_ix], dtype=torch.float32),
        'pupilarea': torch.tensor(dat['pupilarea'][trial_ix], dtype=torch.float32),
        'robs': torch.tensor(robs),
        'dfs': torch.tensor(dfs, dtype=torch.float32),
        'stim': torch.tensor(stim, dtype=torch.float32),
        'tents': torch.tensor(tents, dtype=torch.float32),
        'indices': torch.tensor(np.arange(len(trial_ix)), dtype=torch.int64)}

    ds = GenericDataset(data, device=device)

    return ds, dat

def fit_latents(model, mod1, train_dl, fit_sigmas=False, max_iter=10, seed=None):

    if hasattr(model, 'gain_mu'):
        model.readout_gain.weight.data[:] = 1 #torch.rand(model.readout_gain.weight.shape)
    
    if hasattr(model, 'offset_mu'):
        model.readout_offset.weight.data[:] = 0 #torch.rand(model.readout_offset.weight.shape)
        
    # model.gain_mu.weight.data[:] = res2['zgain']
    # model.gain_mu.weight.data[:] *= 10

    # model.gain_mu.weight_scale = 1.0
    # model.readout_gain.weight_scale = 1.0
    # model.readout_gain.weight.data[:] = 1

    model.drift.weight.data = mod1.drift.weight.data[:,model.cids].clone()
    model.stim.weight.data = mod1.stim.weight.data[:,model.cids].clone()
    model.bias.data = mod1.bias.data[model.cids].clone()

    model.stim.weight.requires_grad = False
    model.drift.weight.requires_grad = True
    model.bias.requires_grad = True

    tol = 1e-6
    for itr in range(max_iter):
        if hasattr(model, 'gain_mu'):
            model.logvar_g.data[:] = 1
            model.logvar_g.requires_grad = fit_sigmas
            model.stim.weight.requires_grad = True
            model.gain_mu.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = False

        if hasattr(model, 'offset_mu'):
            model.logvar_h.data[:] = 1
            model.logvar_h.requires_grad = fit_sigmas
            model.offset_mu.weight.requires_grad = True
            model.readout_offset.weight.requires_grad = False
        
        t1 = fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)

        if hasattr(model, 'gain_mu'):
            model.gain_mu.weight.requires_grad = False
            model.stim.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = True
        
        if hasattr(model, 'offset_mu'):
            model.offset_mu.weight.requires_grad = False
            model.readout_offset.weight.requires_grad = True
        
        t2 = fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        print('%d: %.3f, %.3f' % (itr, t1.val_loss_min, t2.val_loss_min))

        if t1.val_loss_min - t2.val_loss_min < tol:
            break

    return t2.val_loss_min, model

def eval_model(mod2, ds, val_ds):

    sample = ds[:]
    mod2 = mod2.to(ds.device)
    rhat = mod2(sample).detach().cpu().numpy()

    # initialize gains and offsets to zero
    zgav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zg = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zhav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
    zh = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

    if hasattr(mod2, 'tents_as_input'):
        if mod2.tents_as_input:
            latent_input = sample['tents']
        else:
            latent_input = sample['robs']

        if hasattr(mod2, 'latent_gain'):
            zg = mod2.latent_gain(latent_input)
            
            zgav = mod2.readout_gain(zg).detach().cpu()
            zg = zg.detach().cpu()
            
        if hasattr(mod2, 'latent_offset'):
            zh = mod2.latent_offset(latent_input)
            zhav = mod2.readout_offset(zh).detach().cpu()
            zh = zh.detach().cpu()
            
    else:
        if hasattr(mod2, 'gain_mu'):
            zg = mod2.gain_mu.weight.detach()
            zgav = mod2.readout_gain(zg).detach().cpu()
            zg = zg.cpu().clone()
        
        if hasattr(mod2, 'offset_mu'):
            zh = mod2.offset_mu.weight.detach()
            zhav = mod2.readout_offset(zh).detach().cpu()
            zh = zh.cpu().clone()
        
    sample = val_ds[:]
    robs_ = sample['robs'].detach().cpu()
    rhat_ = mod2(sample).detach().cpu()

    r2test = rsquared(robs_[:,mod2.cids], rhat_, sample['dfs'][:,mod2.cids].detach().cpu())

    return {'rhat': rhat, 'zgain': zg, 'zoffset': zh, 'zgainav': zgav, 'zoffsetav': zhav, 'r2test': r2test}

def fit_gain_model(nstim, mod1, NC=None, NT=None,
    cids=None, ntents=None, train_dl=None, include_gain=True,
    include_offset=True, replicates=1): 
    from copy import deepcopy
    
    losses = []
    models = []
    for r in range(replicates):
        mod3 = SharedLatentGain(nstim,
                    NC=NC,
                    NT=NT,
                    cids=cids,
                    num_latent=1,
                    num_tents=ntents,
                    include_stim=True,
                    include_gain=include_gain,
                    include_offset=include_offset,
                    tents_as_input=False,
                    output_nonlinearity='Identity',
                    stim_act_func='lin',
                    stim_reg_vals={'l2':0.01},
                    reg_vals={'d2t': .01, 'l2': 0.001},
                    readout_reg_vals={'l2': .1})

        loss, model = fit_latents(mod3, mod1, train_dl, fit_sigmas=True, max_iter=10, seed=None)
        losses.append(loss)
        models.append(model)
        print('Fit run %d: %.3f' % (r, loss))

    id = np.argmin(np.asarray(losses))
    mod2 = deepcopy(models[id])
    return mod2

def fit_session(fpath,
        apath,
        fname,
        aname,
        ntents=5,
        seed=1234):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    figs = [] # initialize list of figures to save into single pdf
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    ds, dat = get_data(fpath, fname, ntents, device)

    trial_ix = np.where(~np.isnan(dat['gdirection']))[0]
    direction = dat['gdirection'][trial_ix]
    directions = np.unique(direction)
    freq = dat['gfreq'][trial_ix]
    freqs = np.unique(freq)
    nfreq = len(freqs)
    ndir = len(directions)
    
    sample = ds[:]
    nstim = sample['stim'].shape[1]
    NC = sample['robs'].shape[1]
    sx = int(np.ceil(np.sqrt(NC)))
    sy = int(np.round(np.sqrt(NC)))

    TC = sample['stim'].T @ sample['robs']
    TC = (TC.T / sample['stim'].sum(dim=0)).T

    figs.append(plt.figure(figsize=(10,10)))
    for cc in range(NC):
        plt.subplot(sx, sy, cc+1)
        tc = TC[:,cc].detach().cpu().numpy()
        f = plt.plot(directions, np.reshape(tc, [nfreq, ndir]).T, '-o')

    '''
    Build Train / Test sets
    '''
    train_dl, val_dl, indices = get_dataloaders(ds, batch_size=64, use_dropout=True)
    train_inds = indices[0]
    val_inds = indices[1]

    '''
    Baseline model: has no stimulus, can capture slow drift in firing rate for each unit using b0-splines
    '''
    mod0 = SharedGain(nstim,
            NC=NC,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_stim=False,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            latent_noise=True,
            stim_act_func='lin',
            stim_reg_vals={'l2':0.01},
            reg_vals={'l2':0.01},
            act_func='lin')

    print("Fitting baseline model")
    t1 = fit_model(mod0, train_dl, val_dl, use_lbfgs=True, verbose=0)
    print("Done")

    '''
    Model with stimulus and slow drift. Use this to fine which units are driven by the stimulus
    '''
    
    mod1 = SharedGain(nstim,
            NC=NC,
            cids=None,
            num_latent=1,
            num_tents=ntents,
            include_stim=True,
            include_gain=False,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            stim_act_func='lin',
            stim_reg_vals={'l2':1},
            reg_vals={'l2':0.001},
            act_func='lin')

    mod1.drift.weight.data = mod0.drift.weight.data.clone()

    print("Fitting Stimulus Model")
    fit_model(mod1, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("Done")

    res0 = eval_model(mod0, ds, val_dl.dataset)
    res1 = eval_model(mod1, ds, val_dl.dataset)

    figs.append(plt.figure())
    plt.plot(res0['r2test'], 'o', label='Baseline')
    plt.plot(res1['r2test'], 'o', label='Stimulus')
    plt.axhline(0, color='k')
    plt.ylabel('$r^2$')
    plt.xlabel("Unit ID")
    plt.ylim([-0.1,1])
    plt.legend()
    

    '''
    Affine model
    '''
    # cids = np.where(np.logical_and(res1['r2test'] > res0['r2test'], res1['r2test'] > 0))[0]
    cids = np.where(res1['r2test'] > res0['r2test'])[0]
    print("Found %d /%d units with significant stimulus effect" %(len(cids), len(res1['r2test'])))
    
    print("Fitting Affine Model")
    mod2 = fit_gain_model(nstim, mod1, NC=NC, NT=len(ds),
    cids=cids, ntents=ntents, train_dl=train_dl, include_gain=True,
    include_offset=True)

    print("Done")

    ''' 
    Gain Only
    '''
    print("Fitting Gain Model")
    mod3 = fit_gain_model(nstim, mod1, NC=NC, NT=len(ds),
    cids=cids, ntents=ntents, train_dl=train_dl, include_gain=True,
    include_offset=False)

    print("Done")

    ''' 
    Offset Only
    '''
    
    print("Fitting Offset Model")
    mod4 =  fit_gain_model(nstim, mod1, NC=NC, NT=len(ds),
    cids=cids, ntents=ntents, train_dl=train_dl, include_gain=False,
    include_offset=True)

    print("Done")

    sample = ds[:]

    pupil = sample['pupilarea'].detach().cpu().numpy()
    running = sample['runningspeed'].detach().cpu().numpy()
    robs = sample['robs'].detach().cpu().numpy()

    das = dict()
    das['data'] = {'direction': direction,
        'frequency': freq,
        'robs': robs,
        'pupil': pupil, 'running': running, 'train_inds': train_inds,
        'val_inds': val_inds}
    
    '''Evaluate Models'''
    # eval model 0 (Baseline)
    mod0.cids = cids
    mod0.bias.data = mod0.bias.data[cids]
    mod0.drift.weight.data = mod0.drift.weight.data[:,cids]
    mod0.drift.bias.data = mod0.drift.bias.data[cids]
    moddict0 = eval_model(mod0, ds, val_dl.dataset)
    moddict0['model'] = mod0
    das['drift'] = moddict0

    # eval model 1 (Stimulus)
    mod1.cids = cids
    mod1.bias.data = mod1.bias.data[cids]
    mod1.drift.weight.data = mod1.drift.weight.data[:,cids]
    mod1.stim.weight.data = mod1.stim.weight.data[:,cids]
    mod1.drift.bias.data = mod1.drift.bias.data[cids]
    mod1.stim.bias.data = mod1.stim.bias.data[cids]
    moddict1 = eval_model(mod1, ds, val_dl.dataset)
    moddict1['model'] = mod1
    das['stimdrift'] = moddict1


    '''Affine model'''
    moddict2 = eval_model(mod2, ds, val_dl.dataset)
    moddict2['model'] = mod2
    das['affine'] = moddict2

    ''' Gain model'''
    moddict3 = eval_model(mod3, ds, val_dl.dataset)
    moddict3['model'] = mod3
    das['gain'] = moddict3

    ''' Offset model'''
    moddict4 = eval_model(mod4, ds, val_dl.dataset)
    moddict4['model'] = mod4
    das['offset'] = moddict4

    print("Fitting CV PCA")
    ##%% redo PCA on fit neurons
    das['cvpca'] = []
    train_err= []
    test_err = []
    ranks = range(1, 25)
    for rnk in ranks:
        data = ds.covariates['robs'][:,cids]
        Mtrain = train_dl.dataset[:]['dfs'][:,cids]>0
        Mtest = val_dl.dataset[:]['dfs'][:,cids]>0

        U, Vt, tre, te = cv_pca(data, rank=rnk, Mtrain=Mtrain, Mtest=Mtest)
        
        resid = U@Vt - data
        mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

        total_err = data - mu

        tre = 1 - torch.sum(resid**2*Mtrain, dim=0) / torch.sum(total_err**2*Mtrain, dim=0)
        te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

        das['cvpca'].append({'rank': rnk, 'U': U.cpu().numpy(), 'Vt': Vt.cpu().numpy(), 'r2train': tre.cpu().numpy(), 'r2test': te.cpu().numpy()})
        train_err.append((rnk, tre.mean().item()))
        test_err.append((rnk, te.mean().item()))
        
    

    print("Saving...")
    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)
    print("Done")

    ## %% Compare PCA to latent model
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    ax.plot(*list(zip(*train_err)), 'o-b', label='PCA Train Data')
    ax.plot(*list(zip(*test_err)), 'o-r', label='PCA Test Data')
    ax.plot(1, das['stimdrift']['r2test'].mean(), 'o', label='Stim Model', color='k')
    ax.plot(1, das['gain']['r2test'].mean(), 'o', label='1 Gain Latent Model', color='m')
    ax.axhline(das['gain']['r2test'].mean(),color='m')
    # ax.plot(1, res2['r2test'].mean(), 'o', label='1 Gain Autoencoder Model', color='g')
    ax.set_ylabel('Var. Explained')
    ax.set_xlabel('Number of PCs')
    ax.set_title('PCA vs. 1D Shared Gain')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.ylim(0, 1)
    fig.tight_layout()

    

    return das
    

# if main
if __name__ == '__main__':

    fit_session(**vars)

