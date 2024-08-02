
from copy import deepcopy
import os, sys

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

from NDNT.training import LBFGSTrainer

from torch.utils.data import Subset, DataLoader

from models import SharedGain, SharedLatentGain, GenericDataset

'''
Model fitting procedure for the shared gain / offset model

'''

def get_dataloaders(ds, folds=5, batch_size=64, use_dropout=True, seed=1234):
    
    np.random.seed(seed)

    NT = len(ds)

    if use_dropout: # training and test set are the same, but differ in their datafilters
        from copy import deepcopy
        train_ds = deepcopy(ds)
        val_ds = deepcopy(ds)
        test_ds = deepcopy(ds)
        NT, NC = ds.covariates['robs'].shape
        p_holdout = 1/folds

        TrainMask = np.random.rand(NT, NC) > p_holdout
        i,j = np.where(~TrainMask)
        ival = np.random.rand(len(i)) < .5
        ValMask = np.zeros(TrainMask.shape, dtype=bool)
        TestMask = np.zeros(TrainMask.shape, dtype=bool)
        ValMask[i[ival], j[ival]] = True
        TestMask[i[~ival], j[~ival]] = True


        # randomly select some trials for validation
        train_ds.covariates['dfs'] = torch.tensor(np.logical_and(train_ds.covariates['dfs'].cpu().numpy(), TrainMask), dtype=torch.float32, device=ds.device)
        val_ds.covariates['dfs'] = torch.tensor(np.logical_and(val_ds.covariates['dfs'].cpu().numpy(), ValMask), dtype=torch.float32, device=ds.device)
        test_ds.covariates['dfs'] = torch.tensor(np.logical_and(test_ds.covariates['dfs'].cpu().numpy(), TestMask), dtype=torch.float32, device=ds.device)
        
        # build latent datafilters (so you always use the training set to get the latent)
        train_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        val_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        test_ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        ds.covariates['latentdfs'] = train_ds.covariates['dfs'].clone()
        ds.cov_list = list(ds.covariates.keys())

        train_inds = TrainMask
        val_inds = ValMask
        test_inds = TestMask
    
    else: # conventional training and test set
        n_val = NT//folds
        val_inds = np.random.choice(range(NT), size=n_val, replace=False)
        train_inds = np.setdiff1d(range(NT), val_inds)
        ival = np.random.rand(len(val_inds)) < .5
        test_inds = val_inds[~ival]
        val_inds = val_inds[ival]

        train_ds = Subset(ds, train_inds.tolist())
        val_ds = Subset(ds, val_inds.tolist())
        test_ds = Subset(ds, test_inds.tolist())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl, test_dl, (train_inds, val_inds, test_inds)
        
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
    seed=None,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from NDNT.training import Trainer, EarlyStopping, LBFGSTrainer
    model.prepare_regularization()

    if use_lbfgs:
        optimizer = torch.optim.LBFGS(model.parameters(),
                    history_size=100,
                    max_iter=max_iter,
                    tolerance_change=1e-9,
                    line_search_fn='strong_wolfe',
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
        normalize_robs=False,
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
    
    from scipy.ndimage import uniform_filter
    robs_smooth = uniform_filter(robs, size=(50, 1), mode='reflect')
    mx = np.max(robs_smooth, axis=0)
    mn = np.min(robs_smooth, axis=0)

    # good = mx > 1 # max firing rate in sliding window > 1 spike per trial on average
    # pthresh = np.percentile(robs, (0, 99), axis=0)
    # dfs = np.logical_and(robs >= pthresh[0], robs < pthresh[1])
    mu = np.mean(robs, axis=0)
    adiff = np.abs(robs - mu)
    mad = np.median(adiff)
    dfs = (adiff / mad) < 8

    # good = np.mean(dfs, axis=0) > .9 
    good = np.mean(dfs, axis=0) > .8
    # good = np.logical_and(good, mx > 0)
    print("good units %d" %np.sum(good))
    robs = robs[:,good]
    dfs = dfs[:,good]
    
    s = np.std(robs, axis=0)
    mu = np.mean(robs, axis=0)

    mn = robs.min(axis=0)
    mx = robs.max(axis=0)

    if normalize_robs==2:
        # robs = robs / mu
        robs = (robs - mn) / (mx - mn)

    elif normalize_robs==3:
        mu = np.mean(robs, axis=0)
        mad = np.median(np.abs(robs - mu))
        robs = (robs - mu) / mad

    elif normalize_robs==1:
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

def initialize_from_model(model, mod1, train_dl, fit_sigmas=False):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # used for initialization
    data = train_dl.dataset[:]
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    from copy import deepcopy

    model.to(device)
    

    if mod1 is not None:
        robs = data['robs'][:,mod1.cids]
        if 'latentdfs' in data:
            robs = robs * data['latentdfs'][:,mod1.cids]

        robs = robs.to(device)
        
        mod1.to(device)
    
        if hasattr(model, 'gain_mu'):
            
            if hasattr(mod1, 'latent_gain'):
                zg = mod1.latent_gain(robs).detach().clone().to(device)
                zgwts = mod1.readout_gain.weight.detach().clone().cpu().to(device)

                v = zg.var(dim=0)
                # v = 1

                s = (zgwts > 0 ).sum() / zgwts.shape[1]
                if s < .5:
                    zg *= -1
                    zgwts *=-1

                model.gain_mu.weight.data[:] = zg/v
                model.readout_gain.weight.data[:] = zgwts*v

            else: # random initialization
                model.gain_mu.weight.data[:] = 0#torch.rand(model.gain_mu.weight.shape)
                model.readout_gain.weight.data[:] = 1
            
            model.logvar_g.data[:] = 1
            model.logvar_g.requires_grad = fit_sigmas

        if hasattr(model, 'offset_mu'):
            if hasattr(mod1, 'latent_offset'):
                zh = mod1.latent_offset(robs).detach().clone()
                zhwts = mod1.readout_offset.weight.detach().cpu().clone().to(device)
                v = zh.var(dim=0)
                # v = 1
                
                s = (zhwts > 0 ).sum() / zhwts.shape[1]
                if s < .5:
                    zh *= -1
                    zhwts *=-1
                
                model.offset_mu.weight.data[:] = zh.clone().to(device)/v
                model.readout_offset.weight.data[:] = zhwts*v
            else:
                model.offset_mu.weight.data[:] = 0#torch.randn(model.offset_mu.weight.shape)
                model.readout_offset.weight.data[:] = 1
            
            model.logvar_h.data[:] = 1
            model.logvar_h.requires_grad = fit_sigmas

        if model.drift is not None:
            if len(mod1.cids) == len(model.cids):
                model.drift.weight.data = mod1.drift.weight.data.clone()
            else:
                model.drift.weight.data = mod1.drift.weight.data[:,model.cids].clone()
            model.drift.weight.requires_grad = False
            model.bias.requires_grad = True
        else:
            if len(mod1.cids) == len(model.cids):
                model.bias.data[:] = mod1.bias.data.clone()
            else:
                model.bias.data[:] = mod1.bias.data[model.cids].clone()
            model.bias.requires_grad = True

        if len(mod1.cids) == len(model.cids):
            model.stim.weight.data = mod1.stim.weight.data.clone()
            model.bias.data[:] = mod1.bias.data.clone()
        else:
            model.stim.weight.data = mod1.stim.weight.data[:,model.cids].clone()
            model.bias.data[:] = mod1.bias.data[model.cids].clone()

    model.stim.weight.requires_grad = False
    
    return model

def fit_autoencoder(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=10, seed=None):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # data used for validation (sets stopping rule)
    vdata = val_dl.dataset[:]
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)
        
    
    from copy import deepcopy
    
    tol = 1e-9
    model.training = False
    r2 = model_rsquared(model.to(device), vdata)
    l0 = r2.mean().item()

    # l0 = model.validation_step(vdata)['loss'].item()
    model0 = deepcopy(model)
    
    print("Initial: %.4f" %l0)

    if max_iter == 0:
        return l0, model

    # initialize fit by fixing stim, readout, fit gain / offset latents
    model.stim.weight.requires_grad = False

    if hasattr(model, 'latent_gain'):
        model.latent_gain.weight.requires_grad = True
        model.readout_gain.weight.requires_grad = True

    if hasattr(model, 'latent_offset'):
        model.latent_offset.weight.requires_grad = True
        model.readout_offset.weight.requires_grad = True
    
    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)

    r2 = model_rsquared(model, vdata)
    l1 = r2.mean().item()
    
    print('Fit latents: %.4f, %.4f' % (l0, l1))

    # fit iteratively
    for itr in range(max_iter):
        
        if itr > min_iter:
            fit_sigmas = True
            

        # fit stimulus
        if hasattr(model, 'latent_gain'):
            model.latent_gain.weight.requires_grad = False
            model.readout_gain.weight.requires_grad = False
        
        if hasattr(model, 'latent_offset'):
            model.latent_offset.weight.requires_grad = False
            model.readout_offset.weight.requires_grad = False
        
        model.stim.weight.requires_grad = True
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit stim: %.4f, %.4f' % (itr, l0, l1))
        
        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)


        # refit latents
        model.stim.weight.requires_grad = False
        
        if hasattr(model, 'latent_gain'):
            model.latent_gain.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = True

        if hasattr(model, 'latent_offset'):
            model.latent_offset.weight.requires_grad = True
            model.readout_offset.weight.requires_grad = True
        
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit latents: %.4f, %.4f' % (itr, l0, l1))

        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)
    
    return l0, model0


def fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=10, seed=None, fix_readout_weights=False):
    '''
    fit latent variable model given an initialization
    model: the model to be fit
    mod1: the initialization model. If it is an autoencoder, use the autoencoder parameters for the initial condition
    '''
    
    # used for initialization
    data = train_dl.dataset[:]
    vdata = val_dl.dataset[:]
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)
    

    
    
    from copy import deepcopy
    
    tol = 1e-9
    model.training = False
    r2 = model_rsquared(model.to(device), vdata)
    l0 = r2.mean().item()

    # l0 = model.validation_step(vdata)['loss'].item()
    model0 = deepcopy(model)

    # initialization value saved until end in case fitting latents does worse
    l00 = deepcopy(l0)
    model00 = deepcopy(model)
    
    print("Initial: %.4f" %l0)

    if max_iter == 0:
        return l0, model

    # initialize fit by fixing stim, readout, fit gain / offset latents

    model.stim.weight.requires_grad = False

    if hasattr(model, 'gain_mu'):
        model.logvar_g.data[:] = 1
        model.logvar_g.requires_grad = fit_sigmas
        model.gain_mu.weight.requires_grad = True
        model.readout_gain.weight.requires_grad = True

    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)

    if hasattr(model, 'offset_mu'):
        model.logvar_h.data[:] = 1
        model.logvar_h.requires_grad = fit_sigmas
        model.offset_mu.weight.requires_grad = True
        model.readout_offset.weight.requires_grad = True
    
    fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
    r2 = model_rsquared(model, vdata)
    l1 = r2.mean().item()
    
    print('Fit latents: %.4f, %.4f' % (l0, l1))

    # fit iteratively
    for itr in range(max_iter):
        
        if itr > min_iter:
            fit_sigmas = True
            
        # # fit readout weights
        # if not fix_readout_weights:
        #     if hasattr(model, 'gain_mu'):
        #         model.gain_mu.weight.requires_grad = False
        #         model.readout_gain.weight.requires_grad = True
        #         model.logvar_g.data[:] = 0
        #         model.logvar_g.requires_grad = False
            
        #     if hasattr(model, 'offset_mu'):
        #         model.offset_mu.weight.requires_grad = False
        #         model.readout_offset.weight.requires_grad = True
        #         model.logvar_h.data[:] = 0
        #         model.logvar_h.requires_grad = False
            
        #     fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        #     r2 = model_rsquared(model, vdata)
        #     l1 = r2.mean().item()

        #     print('%d) fit loadings: %.4f, %.4f' % (itr, l0, l1))

        #     if itr > min_iter and (l1 - l0) < tol:
        #         print("breaking because tolerance was hit")
        #         break
        #     else:
        #         l0 = l1
        #         model0 = deepcopy(model)
        

        # fit stimulus
        if hasattr(model, 'gain_mu'):
            model.gain_mu.weight.requires_grad = False
            model.readout_gain.weight.requires_grad = False
            model.logvar_g.data[:] = 0
            model.logvar_g.requires_grad = False
        
        if hasattr(model, 'offset_mu'):
            model.offset_mu.weight.requires_grad = False
            model.readout_offset.weight.requires_grad = False
            model.logvar_h.data[:] = 0
            model.logvar_h.requires_grad = False
        
        model.stim.weight.requires_grad = True
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit stim: %.4f, %.4f' % (itr, l0, l1))
        
        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)


        # refit latents
        model.stim.weight.requires_grad = False
        
        if hasattr(model, 'gain_mu'):
            model.logvar_g.data[:] = 1
            model.logvar_g.requires_grad = fit_sigmas
            model.gain_mu.weight.requires_grad = True
            model.readout_gain.weight.requires_grad = True

        if hasattr(model, 'offset_mu'):
            model.logvar_h.data[:] = 1
            model.logvar_h.requires_grad = fit_sigmas
            model.offset_mu.weight.requires_grad = True
            model.readout_offset.weight.requires_grad = True
        
        fit_model(model, train_dl, train_dl, use_lbfgs=True, verbose=0, seed=seed)
        r2 = model_rsquared(model, vdata)
        l1 = r2.mean().item()

        print('%d) fit latents: %.4f, %.4f' % (itr, l0, l1))

        if itr > min_iter and (l1 - l0) < tol:
            print("breaking because tolerance was hit")
            break
        else:
            l0 = l1
            model0 = deepcopy(model)
    
    # if l0 > l00:
    #     print("LVM worse than autoencoder. reverting")
    #     model0 = model00
    #     l0 = l00
    
    return l0, model0

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
            latent_input = sample['robs'][:,mod2.cids] * sample['latentdfs'][:,mod2.cids]

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

def model_rsquared(model, vdata):
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    model.to(device)
    
    for dsub in vdata:
        if vdata[dsub].device != device:
            vdata[dsub] = vdata[dsub].to(device)

    robs_ = vdata['robs'].detach()
    rhat_ = model(vdata).detach()
    model.to('cpu')
    r2 = rsquared(robs_[:,model.cids], rhat_, vdata['dfs'][:,model.cids].detach().cpu())
    return r2

def fit_gain_model(nstim, mod1, NC=None, NT=None,
    cids=None, ntents=None, train_dl=None, val_dl=None,
    include_gain=True,
    include_offset=True,
    max_iter=10,
    num_latent=1,
    verbose=0,
    l2s = [.1],
    d2ts = [0, .001, 0.01, 1]): 
    
    from copy import deepcopy    
    
    if include_gain:
        d2tgs = deepcopy(d2ts)
    else:
        d2tgs = [0]
    
    if include_offset:
        d2ths = deepcopy(d2ts)
    else:
        d2ths = [0]

    losses = []
    models = []
    for l2 in l2s:
        for d2th in d2ths:
            for d2tg in d2tgs:
                mod3 = SharedLatentGain(nstim,
                            NC=NC,
                            NT=NT,
                            cids=cids,
                            num_latent=num_latent,
                            num_tents=ntents,
                            include_stim=True,
                            include_gain=include_gain,
                            include_offset=include_offset,
                            tents_as_input=False,
                            output_nonlinearity='Identity',
                            stim_act_func='lin',
                            stim_reg_vals={'l2':1},
                            gain_reg_vals={'d2t': d2tg, 'BC': {'d2t': 0}},
                            offset_reg_vals={'d2t': d2th, 'BC': {'d2t': 0}},
                            readout_reg_vals={'l2':l2})

                model = initialize_from_model(mod3, mod1, train_dl, fit_sigmas=False)

                loss, model = fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=-1, max_iter=0)
                
                if max_iter == 0:
                    loss = model_rsquared(model, val_dl.dataset[:]).mean().item()
                    losses.append(loss)
                    models.append(model)
                    print('Fit run %.3f,%.3f: %.4f' % (d2tg, d2th, loss))
                    continue

                model.stim.weight.requires_grad = False
                if include_gain:
                    model.gain_mu.weight.requires_grad = True
                    model.readout_gain.weight.requires_grad = True
                    model.logvar_g.data[:] = 1
                    model.logvar_g.requires_grad = False

                if include_offset:
                    model.offset_mu.weight.requires_grad = True
                    model.readout_offset.weight.requires_grad = True
                    model.logvar_h.data[:] = 1
                    model.logvar_h.requires_grad = False

                if ntents > 1:
                    model.drift.weight.requires_grad = False         
                    model.bias.requires_grad = True
                else:
                    model.bias.requires_grad = True

                # fit_model(model, train_dl=train_dl, val_dl=val_dl,
                #     use_lbfgs=True, verbose=verbose)

                # r2 = model_rsquared(model, val_dl.dataset[:])
                # if include_offset:
                #     model.readout_offset.weight.data[:,r2<0] = 0
                
                # if include_gain:
                #     model.readout_gain.weight.data[:,r2<0] = 0                    

                loss, model = fit_latents(model, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=max_iter)

                loss = model_rsquared(model, val_dl.dataset[:]).mean().item()
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                model.to(device)
                train_loss = model.training_step(train_dl.dataset[:])['loss'].item()
                # print(model.gain_mu.reg.vals)
                losses.append(loss)
                models.append(model)
                print('Fit run %.3f,%.3f: %.4f, train loss = %.4f' % (d2tg, d2th, loss, train_loss))

    id = np.argmax(np.asarray(losses))
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

    ds, dat = get_data(fpath, fname, num_tents=ntents, normalize_robs=1)

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
    sample = ds[:]
    NT, nstim = sample['stim'].shape
    NC = sample['robs'].shape[1]
    print("%d Trials n % d Neurons" % (NT, NC))

    # # try to overfit data and throw out outliers
    # mod1 = SharedGain(nstim,
    #             NC=NC,
    #             cids=None,
    #             num_latent=1,
    #             num_tents=ntents,
    #             include_stim=True,
    #             include_gain=False,
    #             include_offset=False,
    #             tents_as_input=False,
    #             output_nonlinearity='Identity',
    #             stim_act_func='elu',
    #             stim_reg_vals={'l2':1},
    #             reg_vals={'l2':0.01},
    #             act_func='lin')

    # from torch.utils.data import DataLoader
    # dl = DataLoader(ds, batch_size=64)
    # mod1.bias.requires_grad = False

    # t0 = fit_model(mod1, dl, dl, use_lbfgs=True, verbose=0)

    # mod1.to(device)
    # rhat = mod1(sample)
    # dfs = (rhat - sample['robs']).detach().cpu().abs() < 20
    # ds.covariates['dfs'] = torch.tensor(dfs.numpy(), dtype=torch.float32).to(device)

    train_dl, val_dl, test_dl, indices = get_dataloaders(ds, batch_size=264, folds=4, use_dropout=True)

    train_inds = indices[0]
    val_inds = indices[1]
    test_inds = indices[2]


    '''
    Step 0: check that the dataset has stable low-dimensional structure at >=4 dimensions
    
    '''
    rnk = 1
    data = ds.covariates['robs']
    Mtrain = train_dl.dataset[:]['dfs']>0
    Mtest = val_dl.dataset[:]['dfs']>0
    U, Vt, tre, te = cv_pca(data, rank=rnk, Mtrain=Mtrain, Mtest=Mtest)
    resid = U@Vt - data
    mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

    total_err = data - mu
    te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)

    cids0 = np.where(te.detach().cpu()>0)[0]
    print("Found %d /%d units with stable low-dimensional structure at rank %d" %(len(cids0), NC, rnk))

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
            reg_vals={'l2':0.001},
            act_func='lin')

    print("Fitting baseline model")
    fit_model(mod0, train_dl, val_dl, use_lbfgs=True, verbose=0)
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

    res0 = eval_model(mod0, ds, test_dl.dataset)
    res1 = eval_model(mod1, ds, test_dl.dataset)

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
    cids = np.where(res1['r2test'] > 0)[0]
    print("Found %d /%d units with significant stimulus + drift model" %(len(cids), len(res1['r2test'])))
    cids = np.union1d(cids0, cids)
    print("Using %d total units for modeling" %len(cids))
    
    print('Fitting Autoencoder version')
    # seed = 1234
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

    """ Fit Offset Autoencoder"""
    mod200 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=False,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')


    if ntents > 1:
        mod200.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod200.bias.requires_grad = False
    else:
        mod200.bias.requires_grad = True

    mod200.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod200.bias.data = mod1.bias.data[cids].clone()
    mod200.stim.weight.requires_grad = False
    mod200.readout_offset.weight_scale = 1.0
    mod200.latent_offset.weight_scale = 1.0
    mod200.readout_offset.weight.data[:] = 1

    mod200.prepare_regularization()

    fit_autoencoder(mod200, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)

    """ Fit Gain Autoencoder"""
    mod201 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=True,
                include_offset=False,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .001},
                act_func='lin')


    if ntents > 1:
        mod201.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod201.bias.requires_grad = False
    else:
        mod201.bias.requires_grad = True

    mod201.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod201.bias.data = mod1.bias.data[cids].clone()
    mod201.stim.weight.requires_grad = False
    mod201.readout_gain.weight_scale = 1.0
    mod201.latent_gain.weight_scale = 1.0
    mod201.readout_gain.weight.data[:] = 1

    mod201.prepare_regularization()

    fit_autoencoder(mod201, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)
    

    """ Fit Affine Autoencoder"""
    mod20 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                latent_noise=False,
                include_stim=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .1},
                act_func='lin')

    if ntents > 1:
        mod20.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod20.drift.weight.requires_grad = True
        mod20.bias.requires_grad = False
    else:
        mod20.bias.requires_grad = True
    mod20.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod20.bias.data = mod1.bias.data[cids].clone()
    mod20.stim.weight.requires_grad = False
    mod20.readout_gain.weight.data[:] = mod201.readout_gain.weight.data.detach().clone()
    mod20.readout_offset.weight.data[:] = mod200.readout_offset.weight.data.detach().clone()
    mod20.latent_gain.weight.data[:] = mod201.latent_gain.weight.data.detach().clone()
    mod20.latent_offset.weight.data[:] = mod200.latent_offset.weight.data.detach().clone()
    
    mod20.prepare_regularization()

    # fit_model(mod20, train_dl, val_dl, use_lbfgs=True, verbose=0, use_warmup=True)
    fit_autoencoder(mod20, train_dl, val_dl, fit_sigmas=False, min_iter=0, max_iter=10)

    mod20.to(device)
    r2 = model_rsquared(mod20, val_dl.dataset[:])
    mod20.readout_gain.weight.data[:,r2<0] = 0
    mod20.readout_offset.weight.data[:,r2<0] = 0
    r2 = model_rsquared(mod20, val_dl.dataset[:])
    l00 = r2.mean().item()
    print('Autoencoder iter %d, val r2: %.4f' %(0, l00))
        # ctr +=1
        # if ctr > 4:
        #     from copy import deepcopy
        #     mod20 = deepcopy(mod1)
        #     break
    res2 = eval_model(mod20, ds, test_dl.dataset)
    print('confirming model r2 = %.4f' %res2['r2test'].mean().item())

    print("Fitting Affine Model")
    mod2 = fit_gain_model(nstim, mod1,
        NC=NC, NT=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.001, 0.01, .1, 1],
        d2ts=[0.00001], #[0.0001, 0.001, .01, .1, 1],
        include_gain=True,
        include_offset=True)

    print("Done")

    ''' 
    Gain Only
    '''
    print("Fitting Gain Model")
    mod3 = fit_gain_model(nstim, mod201, NC=NC, NT=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod2.readout_offset.reg.vals['l2']],
        d2ts=[mod2.gain_mu.reg.vals['d2t']],
        include_gain=True,
        include_offset=False)

    print("Done")

    ''' 
    Offset Only
    '''
    
    print("Fitting Offset Model")
    mod4 =  fit_gain_model(nstim, mod20, NC=NC, NT=len(ds),
        num_latent=1,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[mod2.readout_offset.reg.vals['l2']],
        d2ts=[mod2.offset_mu.reg.vals['d2t']],
        include_gain=False,
        include_offset=True)

    print("convert autoencoder to LVM")
    mod20 = fit_gain_model(nstim, mod20,
        NC=NC, NT=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=True)
    
    mod200 = fit_gain_model(nstim, mod200,
        NC=NC, NT=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=False,
        include_offset=True)
    
    mod201 = fit_gain_model(nstim, mod201,
        NC=NC, NT=len(ds),
        num_latent=1,
        max_iter=0,
        cids=cids, ntents=ntents,
        train_dl=train_dl, val_dl=val_dl,
        verbose=0,
        l2s=[0.01],
        d2ts=[0.01],
        include_gain=True,
        include_offset=False)

    print("Done")

    sample = ds[:]

    pupil = sample['pupilarea'].detach().cpu().numpy()
    running = sample['runningspeed'].detach().cpu().numpy()
    robs = sample['robs'].detach().cpu().numpy()
    dfs = sample['dfs'].detach().cpu().numpy()

    das = dict()
    das['data'] = {'direction': direction,
        'frequency': freq,
        'robs': robs,
        'dfs': dfs,
        'pupil': pupil, 'running': running,
        'train_inds': train_inds,
        'val_inds': val_inds,
        'test_inds': test_inds}
    

    '''Evaluate Models'''
    # eval model 0 (Baseline)
    mod0.cids = cids
    mod0.bias.data = mod0.bias.data[cids]
    mod0.drift.weight.data = mod0.drift.weight.data[:,cids]
    mod0.drift.bias.data = mod0.drift.bias.data[cids]
    moddict0 = eval_model(mod0, ds, test_dl.dataset)
    moddict0['model'] = mod0
    das['drift'] = moddict0

    # eval model 1 (Stimulus)
    mod1.cids = cids
    mod1.bias.data = mod1.bias.data[cids]
    mod1.drift.weight.data = mod1.drift.weight.data[:,cids]
    mod1.stim.weight.data = mod1.stim.weight.data[:,cids]
    mod1.drift.bias.data = mod1.drift.bias.data[cids]
    mod1.stim.bias.data = mod1.stim.bias.data[cids]
    moddict1 = eval_model(mod1, ds, test_dl.dataset)
    moddict1['model'] = mod1
    das['stimdrift'] = moddict1
    
    ''' Autoencoder Offset'''
    moddict200 = eval_model(mod200, ds, test_dl.dataset)
    moddict200['model'] = mod200
    das['offsetae'] = moddict200

    ''' Autoencoder Gain'''
    moddict201 = eval_model(mod201, ds, test_dl.dataset)
    moddict201['model'] = mod201
    das['gainae'] = moddict201

    ''' Autoencoder Affine'''
    moddict20 = eval_model(mod20, ds, test_dl.dataset)
    moddict20['model'] = mod20
    das['affineae'] = moddict20

    '''Affine model'''
    moddict2 = eval_model(mod2, ds, test_dl.dataset)
    moddict2['model'] = mod2
    das['affine'] = moddict2

    # set gain readout weights to zero
    mod2ng = deepcopy(mod2)
    mod2ng.readout_gain.weight.data[:] = 0
    moddict2ng = eval_model(mod2ng, ds, test_dl.dataset)
    moddict2ng['model'] = mod2ng
    das['affine_nogain'] = moddict2ng

    # set offset readout weights to zero
    mod2no = deepcopy(mod2)
    mod2no.readout_offset.weight.data[:] = 0
    moddict2no = eval_model(mod2no, ds, test_dl.dataset)
    moddict2no['model'] = mod2no
    das['affine_nogain'] = moddict2no

    ''' Gain model'''
    moddict3 = eval_model(mod3, ds, test_dl.dataset)
    moddict3['model'] = mod3
    das['gain'] = moddict3

    ''' Offset model'''
    moddict4 = eval_model(mod4, ds, test_dl.dataset)
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
        Mtest = test_dl.dataset[:]['dfs'][:,cids]>0

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

    plot_summary(das, aname)

    return das

def plot_summary(das, aname):
    import matplotlib.gridspec as gridspec
    import matplotlib
    gridspec.GridSpec(2,3)

    model = das['affine']['model']
    zgain = model.gain_mu.get_weights()
    zweight = model.readout_gain.get_weights()
    if np.mean(np.sign(zweight)) < 0: # flip sign if both are negative
        zgain *= -1
        zweight *= -1
    
    zoffset = model.offset_mu.get_weights()
    zoffweight = model.readout_offset.get_weights()
    if np.mean(np.sign(zoffweight)) < 0: # flip sign if both are negative
        zoffset *= -1
        zoffweight *= -1

    robs = das['data']['robs'][:,das['affine']['model'].cids]

    ind = np.argsort(zweight)
    ax0 = plt.subplot2grid((2,3), (0,0), colspan=2)

    plt.imshow(robs[:,ind].T, aspect='auto', interpolation='none', cmap='jet')
    plt.title(aname.replace('.pkl', ''))
    ax0.set_xticklabels([])
    

    ax = plt.subplot2grid((2,3), (1,0), colspan=2)        
    plt.plot(das['data']['running'], 'k', label='Running')
    ax.set_xlim((0, robs.shape[0]))
    # ax = plt.subplot(2,1,2)

    ax2 = ax.twinx()
    plt.plot(zgain, 'r', label='Gain')
    plt.plot(zoffset, 'b', label='Offset')
    ax2.set_xlim((0, robs.shape[0]))
    ax2.legend()
    plt.xlabel("Trial")
    

    from scipy.stats import spearmanr
    rhog = spearmanr(das['data']['running'], zgain)
    rhoo = spearmanr(das['data']['running'], zoffset)

    titlestr = 'Corr w/ running: gain '
    titlestr += "%0.3f" %rhog[0]

    if rhog[1] < 0.05:
        titlestr += "*"

    titlestr += ", offset "
    titlestr += "%0.3f" %rhoo[0]

    if rhoo[1] < 0.05:
        titlestr += "*"
    plt.title(titlestr)

    plt.subplot2grid((2,3), (0,2), colspan=1)
    plt.plot(das['stimdrift']['r2test'], das['affine']['r2test'], 'o')
    plt.plot((0,1), (0,1), 'k')
    plt.xlabel('stim')
    plt.ylabel('affine')

    plt.subplot2grid((2,3), (1,2), colspan=1)
    x = das['affine']['r2test']-das['stimdrift']['r2test']
    x = x.numpy()
    plt.hist(x)
    plt.plot(np.median(x), plt.ylim()[1], 'v')
    plt.xlabel('affine - stim')

# if main
if __name__ == '__main__':

    fit_session(**vars)

