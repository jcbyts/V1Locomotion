
import os, sys
from importlib_metadata import requires

from NDNT.training.trainer import Trainer
sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis

import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn
import geotorch
from scipy.io import loadmat

from datasets import GenericDataset
from NDNT.training import LBFGSTrainer
from NDNT.modules import layers
from NDNT.metrics.mse_loss import MseLoss_datafilter
from torch.utils.data import Subset, DataLoader

class Encoder(nn.Module):
    '''
        Base class for all models.
    '''

    def __init__(self):

        super().__init__()

        self.loss = MseLoss_datafilter()
        self.relu = nn.ReLU()

    def compute_reg_loss(self):
        
        rloss = 0
        if self.stim is not None:
            rloss += self.stim.compute_reg_loss()
        
        if hasattr(self, 'readout_gain'):
            rloss += self.readout_gain.compute_reg_loss()

        if hasattr(self, 'latent_gain'):
            rloss += self.latent_gain.compute_reg_loss()

        if hasattr(self, 'readout_offset'):
            rloss += self.readout_offset.compute_reg_loss()
        
        if hasattr(self, 'latent_offset'):
            rloss += self.latent_offset.compute_reg_loss()
        
        if self.drift is not None:
            rloss += self.drift.compute_reg_loss()
            
        return rloss

    def prepare_regularization(self, normalize_reg = False):
        
        if self.stim is not None:
            self.stim.reg.normalize = normalize_reg
            self.stim.reg.build_reg_modules()

        if hasattr(self, 'readout_gain'):
                self.readout_gain.reg.normalize = normalize_reg
                self.readout_gain.reg.build_reg_modules()
        
        if hasattr(self, 'latent_gain'):
                self.latent_gain.reg.normalize = normalize_reg
                self.latent_gain.reg.build_reg_modules()

        if hasattr(self, 'readout_offset'):
                self.readout_offset.reg.normalize = normalize_reg
                self.readout_offset.reg.build_reg_modules()
        
        if hasattr(self, 'latent_offset'):
                self.latent_offset.reg.normalize = normalize_reg
                self.latent_offset.reg.build_reg_modules()
        
        if self.drift is not None:
            self.drift.reg.normalize = normalize_reg
            self.drift.reg.build_reg_modules()


    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        if 'dfs' in batch:
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)

        regularizers = self.compute_reg_loss()

        return {'loss': loss + regularizers, 'train_loss': loss, 'reg_loss': regularizers}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        if 'dfs' in batch:
            dfs = batch['dfs'][:,self.cids]
            loss = self.loss(y_hat, y, dfs)
        else:
            loss = self.loss(y_hat, y)
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': None}        

class SharedGain(Encoder):

    def __init__(self, stim_dims,
            NC=None,
            cids=None,
            num_latent=5,
            num_tents=10,
            include_stim=True,
            include_gain=True,
            include_offset=True,
            output_nonlinearity='Identity',
            stim_act_func='lin',
            stim_reg_vals={'l2':0.0},
            reg_vals={'l2':0.001},
            act_func='lin'):
        
        super().__init__()

        from copy import deepcopy
        NCTot = deepcopy(NC)
        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)

        self.stim_dims = stim_dims
        self.name = 'LVM'

        if include_stim:
            self.stim = layers.NDNLayer(input_dims=[stim_dims, 1, 1, 1],
                num_filters=NC,
                NLtype=stim_act_func,
                norm_type=0,
                bias=False,
                reg_vals = stim_reg_vals)
        else:
            self.stim = None


        self.bias = nn.Parameter(torch.zeros(NC, dtype=torch.float32))
        self.output_nl = getattr(nn, output_nonlinearity)()

        ''' neuron drift '''
        if num_tents > 1:
            self.drift = layers.NDNLayer(input_dims=[num_tents, 1, 1, 1],
            num_filters=NC,
            NLtype='lin',
            norm_type=0,
            bias=False,
            reg_vals = reg_vals)
        else:
            self.drift = None

        ''' latent variable gain'''
        if include_gain:
            self.latent_gain = layers.NDNLayer(input_dims=[NCTot, 1, 1, 1],
                num_filters=num_latent,
                NLtype='lin',
                norm_type=1,
                bias=False,
                reg_vals = reg_vals)

            self.readout_gain = layers.NDNLayer(input_dims=[num_latent, 1, 1, 1],
                num_filters=NC,
                NLtype='lin',
                norm_type=0,
                bias=False,
                reg_vals = reg_vals)

        ''' latent variable offset'''
        if include_offset:
            self.latent_offset = layers.NDNLayer(input_dims=[NCTot, 1, 1, 1],
                num_filters=num_latent,
                NLtype='lin',
                norm_type=1,
                bias=False,
                reg_vals = reg_vals)

            self.readout_offset = layers.NDNLayer(input_dims=[num_latent, 1, 1, 1],
                num_filters=NC,
                NLtype='lin',
                norm_type=0,
                bias=False,
                reg_vals = reg_vals)

    def forward(self, input):
        
        x = 0
        if self.stim is not None:
            x = x + self.stim(input['stim'])
        
        robs = input['robs']
        if 'dfs' in input:
            robs = robs * input['dfs']

        if hasattr(self, 'latent_gain'):
            zg = self.latent_gain(robs)
            g = self.readout_gain(zg)
            x = x * self.relu(1 + g)
        
        if hasattr(self, 'latent_offset'):
            zh = self.latent_offset(robs)
            h = self.readout_offset(zh)
            x = x + h

        if self.drift is not None:
            x = x + self.drift(input['tents'])

        x = x + self.bias
        x = self.output_nl(x)
        
        return x
        
def rsquared(y, yhat, dfs=None):
    if dfs is None:
        dfs = torch.ones(y.shape, device=y.device)
    ybar = (y * dfs).sum(dim=0) / dfs.sum(dim=0)
    sstot = torch.sum( ( y*dfs - ybar)**2, dim=0)
    ssres = torch.sum( (y*dfs - yhat)**2, dim=0)
    r2 = 1 - ssres/sstot

    return r2.detach().cpu()

def fit_session(fpath, apath, fname, aname, stim_reg_vals={'l2':0.0}, reg_vals={'l2':0.001}):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    NUM_WORKERS = int(os.cpu_count() / 2)

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

    from datasets.utils import tent_basis_generate
    NT = len(freq)
    ntents = 5
    xs = np.linspace(0, NT-1, ntents)
    tents = tent_basis_generate(xs)

    plt.figure()
    plt.plot(tents)
    plt.show()

    robs = dat['robs'][trial_ix,:].astype(np.float32)
    s = np.std(robs, axis=0)
    dfs = (robs / s) < 10

    data = {'runningspeed': torch.tensor(dat['runningspeed'][trial_ix], dtype=torch.float32),
        'pupilarea': torch.tensor(dat['pupilarea'][trial_ix], dtype=torch.float32),
        'robs': torch.tensor(robs),
        'dfs': torch.tensor(dfs, dtype=torch.float32),
        'stim': torch.tensor(stim, dtype=torch.float32),
        'tents': torch.tensor(tents, dtype=torch.float32)}

    ds = GenericDataset(data, device=device)

    sample = ds[:]
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
        # plt.title(flist[isess])
    plt.show()

    '''
    Build Train / Test sets
    '''
    np.random.seed(1234)

    NT = len(ds)
    n_val = NT//5
    val_inds = np.random.choice(range(NT), size=n_val, replace=False)
    train_inds = np.setdiff1d(range(NT), val_inds)

    train_ds = Subset(ds, train_inds.tolist())
    val_ds = Subset(ds, val_inds.tolist())

    batch_size = 64
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    dirname = os.path.join('.', 'checkpoints')
    NBname = 'latents'


    sample = val_ds[:]
    # sample = train_ds[:]

    robs = sample['robs'].cpu()

    plt.figure(figsize=(10,5))
    plt.imshow(robs.T, aspect='auto', interpolation='none')

    #% With LBFGS
    mod0 = SharedGain(stim_dims=nstim, NC=NC, num_tents=ntents, include_stim=False, include_gain=False,
            include_offset=False, output_nonlinearity='Identity',
            stim_act_func='lin', act_func='lin', reg_vals=reg_vals, stim_reg_vals=stim_reg_vals)
    
    mod0.prepare_regularization()

    optimizer = torch.optim.LBFGS(mod0.parameters(),
                    history_size=10,
                    max_iter=10000,
                    tolerance_change=1e-9,
                    line_search_fn=None,
                    tolerance_grad=1e-5)

    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        dirpath=os.path.join(dirname, NBname, 'StimModel'),
        optimize_graph=True,
        log_activations=False,
        set_grad_to_none=False,
        verbose=0)

    trainer.fit(mod0, train_dl.dataset[:], val_dl.dataset[:], seed=1234)
    print("Done fitting stim")

    mod0.to(device)

    # s = mod0.stim(sample['stim'])
    rhat = mod0(sample).detach().cpu().numpy()
    plt.figure()
    r2 = rsquared(robs, rhat)
    plt.plot(r2)
    plt.axhline(0, color='k')
    plt.xlabel('Neuron ID')
    plt.ylabel("r^2")
    plt.show()

    #% try with drift
    mod1 = SharedGain(stim_dims=nstim, NC=NC, num_tents=ntents, include_gain=False,
        include_offset=False, output_nonlinearity='Identity',
        stim_act_func='lin', act_func='lin', reg_vals=reg_vals, stim_reg_vals=stim_reg_vals)

    mod1.prepare_regularization()

    optimizer = torch.optim.LBFGS(mod1.parameters(),
                    history_size=10,
                    max_iter=10000,
                    tolerance_change=1e-9,
                    line_search_fn=None,
                    tolerance_grad=1e-5)

    trainer = LBFGSTrainer(
        optimizer=optimizer,
        device=device,
        dirpath=os.path.join(dirname, NBname, 'StimModelDrift'),
        optimize_graph=True,
        log_activations=False,
        set_grad_to_none=False,
        verbose=0)

    trainer.fit(mod1, train_dl.dataset[:], val_dl.dataset[:], seed=1234)
    print("Done fitting drift")
    mod1.to(device)
    rhat = mod1(sample).detach().cpu().numpy()
    r2_1 = rsquared(robs, rhat)
    plt.plot(r2)
    plt.plot(r2_1)
    plt.axhline(0, color='k')
    plt.xlabel('Neuron ID')
    plt.ylabel("r^2")

    # return mod1, r2_1, r2
    cids = np.where( (r2_1 > r2).numpy())[0]

    # drift_pred = mod1.drift(ds[:]['tents']).detach().cpu()
    # plt.figure(figsize=(10,5))
    # f = plt.plot(drift_pred)

    #% fit gain model
    def fit_gainmodel(mod1,
        nlatent=1,
        ntents=0,
        include_offset=True,
        include_gain=True,
        use_adam=True,
        cids=None):

        mod2 = SharedGain(stim_dims=nstim, NC=NC,
            num_latent=nlatent,
            num_tents=ntents,
            cids=cids,
            act_func='lin',
            include_offset=include_offset,
            include_gain=include_gain,
            stim_act_func='lin',
            output_nonlinearity='ReLU', reg_vals=reg_vals, stim_reg_vals=stim_reg_vals)


        mod2 = mod2.to(device)
        mod2.stim.weight.data = mod1.stim.weight.data.clone()
        mod2.bias.data = mod1.bias.data.clone()
        if ntents > 0:
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
                device=device,
                dirpath=os.path.join(dirname, NBname, 'GainModel'),
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
                device=device,
                dirpath=os.path.join(dirname, NBname, 'GainModel'),
                optimize_graph=True,
                log_activations=False,
                set_grad_to_none=False,
                verbose=0)

            trainer.fit(mod2, train_ds[:], val_ds[:], seed=1234)

        # # unfreeze stimulus weights
        # mod2.stim.weight.requires_grad = True # freeze stimulus model first
        # mod2.stim.bias.requires_grad = True # freeze stimulus model first
        # # mod2.drift.weight.requires_grad = True # freeze stimulus model first

        # trainer.fit(mod2, train_ds[:], val_ds[:])
        mod2.to(device)
        loss = mod2.training_step(train_ds[:])['loss'].item()
        val_loss = mod2.training_step(val_ds[:])['loss'].item()
        return mod2, loss, val_loss

    def train_multistart(nlatent=1, nruns=10, ntents=0, include_offset=True, include_gain=True, cids=cids):

        #% fit 10 runs
        from copy import deepcopy
        mods = []
        losses = []
        val_losses = []

        for i in range(nruns):
            mod2, loss, val_loss = fit_gainmodel(mod1, nlatent=nlatent, ntents=ntents, include_offset=include_offset, include_gain=include_gain, cids=cids)
            mods.append(deepcopy(mod2))
            losses.append(loss)
            val_losses.append(val_loss)

        return mods, losses, val_losses

    def eval_model(mod2, ds, val_ds):

        sample = ds[:]
        mod2 = mod2.to(device)
        rhat = mod2(sample).detach().cpu().numpy()

        if hasattr(mod2, 'latent_gain'):
            zg = mod2.latent_gain(sample['robs'])
           
            zgav = mod2.readout_gain(zg).detach().cpu()
            zg = zg.detach().cpu()
        else:
            zgav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
            zg = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

        if hasattr(mod2, 'latent_offset'):
            zh = mod2.latent_offset(sample['robs'])
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
        

    sample = ds[:]

    pupil = sample['pupilarea'].detach().cpu().numpy()
    running = sample['runningspeed'].detach().cpu().numpy()
    robs = sample['robs'].detach().cpu().numpy()

    das = dict()
    das['data'] = {'direction': direction,
        'frequency': freq,
        'robs': robs,
        'pupil': pupil, 'running': running}
    
    mod0.cids = cids
    mod0.bias.data = mod0.bias.data[cids]
    mod0.drift.weight.data = mod0.drift.weight.data[:,cids]
    mod0.drift.bias.data = mod0.drift.bias.data[cids]
    moddict0 = eval_model(mod0, ds, val_ds)
    moddict0['model'] = mod0
    das['stim'] = moddict0

    mod1.cids = cids
    mod1.bias.data = mod1.bias.data[cids]
    mod1.drift.weight.data = mod1.drift.weight.data[:,cids]
    mod1.stim.weight.data = mod1.stim.weight.data[:,cids]
    mod1.drift.bias.data = mod1.drift.bias.data[cids]
    mod1.stim.bias.data = mod1.stim.bias.data[cids]
    moddict1 = eval_model(mod1, ds, val_ds)
    moddict1['model'] = mod1
    das['stimdrift'] = moddict1

    from copy import deepcopy
    ''' fit Affine model'''
    print("Fitting stim * gain + offset")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=10, ntents=ntents, include_offset=True, include_gain=True, cids=cids)
    bestid = np.nanargmin(vallosses)
    mod4 = deepcopy(models[bestid])
    moddict4 = eval_model(mod4, ds, val_ds)
    moddict4['model'] = mod4
    moddict4['valloss'] = np.asarray(vallosses)
    das['affine'] = moddict4
    print("Done")

    ''' fit offset model'''
    print("Fitting stim + offset")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=5, include_offset=True, include_gain=False)
    bestid = np.nanargmin(vallosses)
    mod2 = deepcopy(models[bestid])
    moddict2 = eval_model(mod2, ds, val_ds)
    moddict2['model'] = mod2
    moddict2['valloss'] = np.asarray(vallosses)
    das['offset'] = moddict2
    print("Done")

    ''' fit gain model'''
    print("Fitting stim * gain")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=5, include_offset=False, include_gain=True)
    bestid = np.nanargmin(vallosses)
    mod3 = deepcopy(models[bestid])
    moddict3 = eval_model(mod3, ds, val_ds)
    moddict3['model'] = mod3
    moddict3['valloss'] = np.asarray(vallosses)
    das['gain'] = moddict3
    print("Done")

    nogain_inds = np.where(das['affine']['r2test'] < das['offset']['r2test'])[0]
    mod5 = deepcopy(das['affine']['model'])
    mod5.readout_gain.weight.data[:] = 0
    # mod5.stim.weight.data[nogain_inds,:] = mod2.stim.weight.data[nogain_inds,:].clone()
    # mod5.readout_gain.linear.weight.data[nogain_inds] = 0
    moddict5 = eval_model(mod5, ds, val_ds)
    moddict5['model'] = mod5
    moddict5['valloss'] = np.nan
    das['affineadjust'] = moddict5

    # # pickle dataset
    import pickle

    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)

# if main
if __name__ == '__main__':

    fit_session(**vars)

