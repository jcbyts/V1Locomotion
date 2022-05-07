
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

from models import SharedGain

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

def fit_session(fpath,
        apath,
        fname,
        aname,
        ntents=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = 1234
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

    plt.figure(figsize=(10,10))
    for cc in range(NC):
        plt.subplot(sx, sy, cc+1)
        tc = TC[:,cc].detach().cpu().numpy()
        f = plt.plot(directions, np.reshape(tc, [nfreq, ndir]).T, '-o')
    plt.show()

    '''
    Build Train / Test sets
    '''
    train_dl, val_dl, indices = get_dataloaders(ds, batch_size=64)
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
            stim_act_func='lin',
            stim_reg_vals={'l2':0.00},
            reg_vals={'l2':0.00},
            act_func='lin')

    mod0.prepare_regularization()

    print("Fitting baseline model")
    fit_model(mod0, train_dl, val_dl, verbose=0, use_lbfgs=True)
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
            stim_reg_vals={'l2':0.01},
            reg_vals={'l2':1},
            act_func='lin')

    mod1.drift.weight.data = mod0.drift.weight.data.clone()
    mod1.prepare_regularization()

    print("Fitting Stimulus Model")
    fit_model(mod1, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("Done")

    res0 = eval_model(mod0, ds, val_dl.dataset)
    res1 = eval_model(mod1, ds, val_dl.dataset)

    plt.figure()
    plt.plot(res0['r2test'], 'o', label='Baseline')
    plt.plot(res1['r2test'], 'o', label='Stimulus')
    plt.axhline(0, color='k')
    plt.ylabel('$r^2$')
    plt.xlabel("Unit ID")
    plt.ylim([-0.1,1])

    '''
    Affine model
    '''
    cids = np.where(np.logical_and(res1['r2test'] > res0['r2test'], res1['r2test'] > 0))[0]

    mod2 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                include_stim=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2':0.01},
                reg_vals={'l2':1},
                act_func='lin')

    mod2.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
    mod2.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod2.bias.data = mod1.bias.data[cids].clone()
    mod2.drift.weight.requires_grad = False # individual neuron drift is fixed
    mod2.stim.weight.requires_grad = True # stimulus fit at same time as gain
    mod2.bias.requires_grad = True

    mod2.prepare_regularization()

    print("Fitting Affine Model")
    fit_model(mod2, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("Done")

    ''' 
    Gain Only
    '''
    mod3 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                include_stim=True,
                include_gain=True,
                include_offset=False,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2':0.01},
                reg_vals={'l2':1},
                act_func='lin')

    mod3.drift.weight.data = mod2.drift.weight.data.clone()
    mod3.stim.weight.data = mod2.stim.weight.data.clone()
    mod3.bias.data = mod2.bias.data.clone()
    mod3.latent_gain.weight.data = mod2.latent_gain.weight.data.clone()
    mod3.readout_gain.weight.data = mod2.readout_gain.weight.data.clone()
    mod3.drift.weight.requires_grad = True
    mod3.stim.weight.requires_grad = True
    mod3.bias.requires_grad = True

    mod3.prepare_regularization()

    print("Fitting Gain Model")
    fit_model(mod3, train_dl, val_dl, verbose=0, use_lbfgs=True)
    print("Done")

    ''' 
    Offset Only
    '''
    mod4 = SharedGain(nstim,
                NC=NC,
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                include_stim=True,
                include_gain=False,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2':0.01},
                reg_vals={'l2':1},
                act_func='lin')

    mod4.drift.weight.data = mod2.drift.weight.data.clone()
    mod4.stim.weight.data = mod2.stim.weight.data.clone()
    mod4.bias.data = mod2.bias.data.clone()
    mod4.latent_offset.weight.data = mod2.latent_offset.weight.data.clone()
    mod4.readout_offset.weight.data = mod2.readout_offset.weight.data.clone()
    mod4.drift.weight.requires_grad = False
    mod4.stim.weight.requires_grad = True
    mod4.bias.requires_grad = True

    mod4.prepare_regularization()

    print("Fitting Offset Model")
    fit_model(mod4, train_dl, val_dl, verbose=0, use_lbfgs=True)
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

    print("Saving...")
    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)
    print("Done")

    return das
    

# if main
if __name__ == '__main__':

    fit_session(**vars)

