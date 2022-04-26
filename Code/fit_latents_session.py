
import os, sys
from importlib_metadata import requires

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

def get_dataloaders(ds, folds=5, batch_size=64):
    np.random.seed(1234)

    NT = len(ds)
    n_val = NT//folds
    val_inds = np.random.choice(range(NT), size=n_val, replace=False)
    train_inds = np.setdiff1d(range(NT), val_inds)

    train_ds = Subset(ds, train_inds.tolist())
    val_ds = Subset(ds, val_inds.tolist())

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    return train_dl, val_dl
        
def rsquared(y, yhat, dfs=None):
    if dfs is None:
        dfs = torch.ones(y.shape, device=y.device)
    ybar = (y * dfs).sum(dim=0) / dfs.sum(dim=0)
    sstot = torch.sum( ( y*dfs - ybar)**2, dim=0)
    ssres = torch.sum( (y*dfs - yhat)**2, dim=0)
    r2 = 1 - ssres/sstot

    return r2.detach().cpu()

def fit_model(model, train_dl, val_dl,
    verbose=1,
    unit_weight=False,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from NDNT.training import Trainer, EarlyStopping

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.01,
        steps_per_epoch=len(train_dl), epochs=20000)

    earlystopping = EarlyStopping(patience=500, verbose=False)

    trainer = Trainer(model, optimizer,
            scheduler,
            device=device,
            optimize_graph=True,
            max_epochs=20000,
            early_stopping=earlystopping,
            log_activations=False,
            scheduler_after='batch',
            scheduler_metric=None,
            verbose=verbose)

    if unit_weight:
        r = 0
        for data in train_dl:
            r += data['robs'].sum(dim=0)

        r = r.cpu()    
        model.loss.unit_weighting = True
        model.loss.unit_weights = (1/r) / (1/r).sum()

    trainer.fit(model, train_dl, val_dl)

def get_data(fpath, fname, num_tents=10,
        normalize_robs=True,
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
    dfs = np.abs( (robs - mu) / s) < 10 # filter out outliers
    if normalize_robs:
        robs = ( (robs-mu) / s)

    data = {'runningspeed': torch.tensor(dat['runningspeed'][trial_ix], dtype=torch.float32),
        'pupilarea': torch.tensor(dat['pupilarea'][trial_ix], dtype=torch.float32),
        'robs': torch.tensor(robs),
        'dfs': torch.tensor(dfs, dtype=torch.float32),
        'stim': torch.tensor(stim, dtype=torch.float32),
        'tents': torch.tensor(tents, dtype=torch.float32)}

    ds = GenericDataset(data, device=device)

    return ds, dat

#% fit gain model
def fit_gainmodel_old(mod1,
    train_dl,
    val_dl,
    ds=None,
    nlatent=1,
    ntents=0,
    include_offset=True,
    include_gain=True,
    tents_as_input=False,
    use_adam=True,
    reg_vals={'l2': 0.001},
    stim_reg_vals={'l2': 0.001},
    cids=None):

    assert ds is not None, "Must provide dataset"

    nstim, NC, = mod1.stim.weight.shape

    mod2 = SharedGain(stim_dims=nstim, NC=NC,
        num_latent=nlatent,
        num_tents=ntents,
        cids=cids,
        act_func='lin',
        tents_as_input=tents_as_input,
        include_offset=include_offset,
        include_gain=include_gain,
        stim_act_func='lin',
        output_nonlinearity='ReLU', reg_vals=reg_vals, stim_reg_vals=stim_reg_vals)


    mod2 = mod2.to(ds.device)
    mod2.stim.weight.data = mod1.stim.weight.data.clone()
    mod2.bias.data = mod1.bias.data.clone()
    if mod2.drift is not None:
        mod2.drift.weight.data = mod1.drift.weight.data.clone()
        mod2.drift.weight.requires_grad = False # freeze stimulus model first
    # mod2.drift.weight.data[:] = 0

    mod2.stim.weight.requires_grad = False # freeze stimulus model first
    mod2.bias.requires_grad = False # freeze stimulus model first
    mod2.prepare_regularization()

    #% Fit gain
    if use_adam:
        optimizer = torch.optim.AdamW(mod2.parameters(), lr=1e-3, weight_decay=1e-5)

        trainer = Trainer(
            optimizer=optimizer,
            device=ds.device,
            dirpath=os.path.join('./checkpoints', 'GainModel'),
            optimize_graph=True,
            log_activations=False,
            set_grad_to_none=False,
            verbose=0)
        
        trainer.fit(mod2, train_dl, val_dl, seed=1234)

    else:
        optimizer = torch.optim.LBFGS(mod2.parameters(),
                history_size=10,
                max_iter=10000,
                tolerance_change=1e-9,
                line_search_fn=None,
                tolerance_grad=1e-5)

        trainer = LBFGSTrainer(
            optimizer=optimizer,
            device=ds.device,
            dirpath=os.path.join('./checkpoints', 'GainModel'),
            optimize_graph=True,
            log_activations=False,
            set_grad_to_none=False,
            verbose=0)

        trainer.fit(mod2, train_dl.dataset[:], val_dl.dataset[:], seed=1234)

    # # unfreeze stimulus weights
    # mod2.stim.weight.requires_grad = True # freeze stimulus model first
    # mod2.stim.bias.requires_grad = True # freeze stimulus model first
    # # mod2.drift.weight.requires_grad = True # freeze stimulus model first

    # trainer.fit(mod2, train_ds[:], val_ds[:])
    mod2.to(device)
    loss = mod2.training_step(train_ds[:])['loss'].item()
    val_loss = mod2.training_step(val_ds[:])['loss'].item()
    return mod2, loss, val_loss

def train_multistart(mod1, 
    nlatent=1,
    nruns=10,
    ntents=0,
    include_offset=True,
    include_gain=True, cids=None):

        #% fit 10 runs
        from copy import deepcopy
        mods = []
        losses = []
        val_losses = []

        for i in range(nruns):
            mod2, loss, val_loss = fit_gainmodel(mod1,
                nlatent=nlatent, ntents=ntents,
                include_offset=include_offset,
                include_gain=include_gain,
                cids=cids)

            mods.append(deepcopy(mod2))
            losses.append(loss)
            val_losses.append(val_loss)

        return mods, losses, val_losses

def eval_model(mod2, ds, val_ds):

    sample = ds[:]
    mod2 = mod2.to(ds.device)
    rhat = mod2(sample).detach().cpu().numpy()

    if mod2.tents_as_input:
        latent_input = sample['tents']
    else:
        latent_input = sample['robs']

    if hasattr(mod2, 'latent_gain'):
        zg = mod2.latent_gain(latent_input)
        
        zgav = mod2.readout_gain(zg).detach().cpu()
        zg = zg.detach().cpu()
    else:
        zgav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
        zg = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

    if hasattr(mod2, 'latent_offset'):
        zh = mod2.latent_offset(latent_input)
        zhav = mod2.readout_offset(zh).detach().cpu()
        zh = zh.detach().cpu()
    else:
        zhav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
        zh = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

    sample = val_ds[:]
    robs_ = sample['robs'].detach().cpu()
    rhat_ = mod2(sample).detach().cpu()

    r2test = rsquared(robs_[:,mod2.cids], rhat_)

    return {'rhat': rhat, 'zgain': zg, 'zoffset': zh, 'zgainav': zgav, 'zoffsetav': zhav, 'r2test': r2test}

def fit_session(fpath,
        apath,
        fname,
        aname,
        ntents=5):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
    train_dl, val_dl = get_dataloaders(ds, batch_size=64)

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
    fit_model(mod0, train_dl, val_dl, verbose=0)
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
            stim_reg_vals={'l2':0.00},
            reg_vals={'l2':0.00},
            act_func='lin')

    mod1.drift.weight.data = mod0.drift.weight.data.clone()
    mod1.prepare_regularization()

    print("Fitting Stimulus Model")
    fit_model(mod1, train_dl, val_dl, verbose=0)
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
                stim_reg_vals={'l2':0.00},
                reg_vals={'l2':0.00},
                act_func='lin')

    mod2.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
    mod2.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod2.bias.data = mod1.bias.data[cids].clone()
    mod2.drift.weight.requires_grad = False # individual neuron drift is fixed
    mod2.stim.weight.requires_grad = True # stimulus fit at same time as gain
    mod2.bias.requires_grad = False

    mod2.prepare_regularization()

    print("Fitting Affine Model")
    fit_model(mod2, train_dl, val_dl, verbose=0)
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
                stim_reg_vals={'l2':0.00},
                reg_vals={'l2':0.00},
                act_func='lin')

    mod3.drift.weight.data = mod2.drift.weight.data.clone()
    mod3.stim.weight.data = mod2.stim.weight.data.clone()
    mod3.bias.data = mod2.bias.data.clone()
    mod3.latent_gain.weight.data = mod2.latent_gain.weight.data.clone()
    mod3.readout_gain.weight.data = mod2.readout_gain.weight.data.clone()
    mod3.drift.weight.requires_grad = False
    mod3.stim.weight.requires_grad = True
    mod3.bias.requires_grad = False

    mod3.prepare_regularization()

    print("Fitting Gain Model")
    fit_model(mod3, train_dl, val_dl, verbose=0)
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
                stim_reg_vals={'l2':0.00},
                reg_vals={'l2':0.00},
                act_func='lin')

    mod4.drift.weight.data = mod2.drift.weight.data.clone()
    mod4.stim.weight.data = mod2.stim.weight.data.clone()
    mod4.bias.data = mod2.bias.data.clone()
    mod4.latent_offset.weight.data = mod2.latent_offset.weight.data.clone()
    mod4.readout_offset.weight.data = mod2.readout_offset.weight.data.clone()
    mod4.drift.weight.requires_grad = False
    mod4.stim.weight.requires_grad = True
    mod4.bias.requires_grad = False

    mod4.prepare_regularization()

    print("Fitting Offset Model")
    fit_model(mod4, train_dl, val_dl, verbose=0)
    print("Done")

    sample = ds[:]

    pupil = sample['pupilarea'].detach().cpu().numpy()
    running = sample['runningspeed'].detach().cpu().numpy()
    robs = sample['robs'].detach().cpu().numpy()

    das = dict()
    das['data'] = {'direction': direction,
        'frequency': freq,
        'robs': robs,
        'pupil': pupil, 'running': running}
    
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
    das['offset'] = moddict3

    ''' Offset model'''
    moddict4 = eval_model(mod4, ds, val_dl.dataset)
    moddict4['model'] = mod4
    das['gain'] = moddict4

    print("Saving...")
    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)
    print("Done")

    return das
    

# if main
if __name__ == '__main__':

    fit_session(**vars)

