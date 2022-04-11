
import os, sys
sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis

import matplotlib.pyplot as plt

# Import torch
import torch
from torch import nn
from scipy.io import loadmat

from datasets import GenericDataset
from NDNT.training import LBFGSTrainer
from torch.utils.data import Subset

class Encoder(nn.Module):
    '''
        Base class for all models.
    '''

    def __init__(self):

        super().__init__()

        self.loss = nn.MSELoss()
        self.gamma_stim = 0.01
        self.gamma_latents = 0.01
        self.gamma_orth = 0
        self.gamma_readout = 0.01
        self.relu = nn.ReLU()
        self.register_buffer('reg_placeholder', torch.zeros(1, dtype=torch.float32))

    def training_step(self, batch, batch_idx=None):  # batch_indx not used, right?
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        regpen = self.reg_placeholder
        if hasattr(self, 'stim'):
            regpen = self.gamma_stim * torch.sum(self.stim.weight**2).sqrt()
        if hasattr(self, 'latent_gain'):
            regpen = regpen + self.gamma_latents * torch.sum(self.latent_gain.weight**2).sqrt()
            # # orthogonality penalty on latents
            # z = self.latent_gain(batch['robs'])
            # w = (z.T @ z).abs()
            # orth_pen = w.sum() - w.trace()
            # regpen = regpen + self.gamma_orth * orth_pen
        if hasattr(self, 'latent_offset'):
            regpen = regpen + self.gamma_latents * torch.sum(self.latent_offset.weight**2).sqrt()
            # # orthogonality penalty on latents
            # z = self.latent_offset(batch['robs'])
            # w = (z.T @ z).abs()
            # orth_pen = w.sum() - w.trace()
            # regpen = regpen + self.gamma_orth * orth_pen

        if hasattr(self, 'readout_offset'):
            regpen = regpen + self.gamma_latents * torch.sum(self.readout_offset.linear.weight**2).sqrt()
            regpen = regpen + self.gamma_readout * torch.sum(self.relu(-self.readout_offset.linear.weight))
        if hasattr(self, 'readout_gain'):
            regpen = regpen + self.gamma_latents * torch.sum(self.readout_gain.linear.weight**2).sqrt()
            regpen = regpen + self.gamma_readout * torch.sum(self.relu(-self.readout_gain.linear.weight))
            # regpen = regpen + self.gamma_readout * torch.sum(self.relu(-self.readout_gain.linear.weight)**2).sqrt()

        return {'loss': loss + regpen, 'train_loss': loss, 'reg_loss': regpen}

    def validation_step(self, batch, batch_idx=None):
        
        y = batch['robs'][:,self.cids]

        y_hat = self(batch)

        loss = self.loss(y_hat, y)
        
        return {'loss': loss, 'val_loss': loss, 'reg_loss': self.reg_placeholder}        

class SharedGain(Encoder):

    def __init__(self, stim_dims,
            NC=None,
            cids=None,
            num_latent=5,
            num_tents=10,
            include_gain=True,
            include_offset=True,
            output_nonlinearity='Identity',
            stim_act_func='Identity',
            act_func='ReLU'):
        
        super().__init__()

        if cids is None:
            self.cids = list(range(NC))
        else:
            self.cids = cids
            NC = len(cids)

        self.stim_dims = stim_dims
        self.name = 'LVM'
        self.act_func = getattr(nn, act_func)()
        self.stim_act_func = getattr(nn, stim_act_func)()
        self.gainnl = nn.ReLU()
        self.output_nl = getattr(nn, output_nonlinearity)()

        ''' stimulus processing '''
        self.stim = nn.Linear(stim_dims, NC, bias=True)

        ''' neuron drift '''
        if num_tents > 1:
            self.drift = nn.Linear(num_tents, NC, bias=False)
        else:
            self.drift = None

        ''' latent variable gain'''
        if include_gain:
            self.latent_gain = nn.Linear(NC, num_latent, bias=True)
            # self.latent_gain.weight.data[:] = 1/NC

            self.readout_gain = nn.Sequential()
            self.readout_gain.add_module('linear', nn.Linear(num_latent, NC, bias=True))
            self.readout_gain.linear.weight.data[:] = .1

            if self.act_func is not None:
                self.readout_gain.add_module('act_func', self.act_func)

        ''' latent variable offset'''
        if include_offset:
            self.latent_offset = nn.Linear(NC, num_latent, bias=True)
            # self.latent_offset.weight.data[:] = 1/NC

            self.readout_offset = nn.Sequential()
            self.readout_offset.add_module('linear', nn.Linear(num_latent, NC, bias=True))
            self.readout_offset.linear.weight.data[:] = .1

        # self.output_NL = nn.Softplus()
        # self.bias = nn.Parameter(torch.zeros(NC, dtype=torch.float32))

    def forward(self, input):

        x = self.stim_act_func(self.stim(input['stim']))
        
        if hasattr(self, 'latent_gain'):
            zg = self.latent_gain(input['robs'])
            zg = self.gainnl(zg)
            g = self.readout_gain(zg)
            x = x * (1 + g)
        
        if hasattr(self, 'latent_offset'):
            zh = self.latent_offset(input['robs'])
            h = self.readout_offset(zh)
            x = x + h

        if self.drift is not None:
            x = x + self.drift(input['tents'])
        
        return x
        
def rsquared(y, yhat):
    ybar = y.mean(dim=0)
    sstot = torch.sum( (y - ybar)**2, dim=0)
    ssres = torch.sum( (y - yhat)**2, dim=0)
    r2 = 1 - ssres/sstot

    return r2.detach().cpu()

def fit_session(fpath, apath, fname, aname):

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

    data = {'runningspeed': torch.tensor(dat['runningspeed'][trial_ix], dtype=torch.float32),
        'pupilarea': torch.tensor(dat['pupilarea'][trial_ix], dtype=torch.float32),
        'robs': torch.tensor(dat['robs'][trial_ix,:].astype(np.float32)),
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

    dirname = os.path.join('.', 'checkpoints')
    NBname = 'latents'


    #% With LBFGS
    mod0 = SharedGain(stim_dims=nstim, NC=NC, num_tents=0, include_gain=False,
        include_offset=False, output_nonlinearity='Identity',
        stim_act_func='Identity', act_func='Identity')

    mod0.gamma_stim = .01

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

    trainer.fit(mod0, train_ds[:], val_ds[:], seed=1234)
    print("Done fitting stim")

    #% plot weights
    plt.figure()
    plt.plot(mod0.stim.weight.T.detach().cpu())
    plt.show()
        
    sample = val_ds[:]

    robs = sample['robs'].cpu()

    mod0.to(device)

    s = mod0.stim(sample['stim'])
    rhat = mod0(sample).detach().cpu().numpy()

    r2 = rsquared(robs, rhat)
    plt.figure()
    plt.plot(r2)
    plt.axhline(0, color='k')
    plt.xlabel('Neuron ID')
    plt.ylabel("r^2")
    plt.show()


    #% try with drift
    mod1 = SharedGain(stim_dims=nstim, NC=NC, num_tents=ntents, include_gain=False,
        include_offset=False, output_nonlinearity='Identity',
        stim_act_func='Identity', act_func='Identity')

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

    trainer.fit(mod1, train_ds[:], val_ds[:], seed=1234)
    print("Done fitting drift")
    mod1.to(device)
    rhat = mod1(sample).detach().cpu().numpy()
    r2_1 = rsquared(robs, rhat)
    plt.plot(r2)
    plt.plot(r2_1)
    plt.axhline(0, color='k')
    plt.xlabel('Neuron ID')
    plt.ylabel("r^2")

    # drift_pred = mod1.drift(ds[:]['tents']).detach().cpu()
    # plt.figure(figsize=(10,5))
    # f = plt.plot(drift_pred)

    #% fit gain model
    def fit_gainmodel(mod1,
        nlatent=1,
        include_offset=True,
        include_gain=True):

        mod2 = SharedGain(stim_dims=nstim, NC=NC,
            num_latent=nlatent,
            num_tents=ntents,
            act_func='Identity',
            include_offset=include_offset,
            include_gain=include_gain,
            stim_act_func='Identity',
            output_nonlinearity='Identity')

        mod2.gamma_stim = .1
        mod2.gamma_latents = 1
        mod2.gamma_orth = 0
        mod2.gamma_readout = 1

        mod2 = mod2.to(device)
        mod2.stim.weight.data = mod1.stim.weight.data.clone() * 2 / 3
        mod2.stim.bias.data = mod1.stim.bias.data.clone()
        # mod2.drift.weight.data = mod1.drift.weight.data.clone()
        mod2.drift.weight.data[:] = 0

        mod2.stim.weight.requires_grad = True # freeze stimulus model first
        mod2.stim.bias.requires_grad = True # freeze stimulus model first

        #% Fit gain
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

        # # unfreeze
        # mod2.stim.weight.requires_grad = True # freeze stimulus model first
        # mod2.stim.bias.requires_grad = True # freeze stimulus model first

        # trainer.fit(mod2, train_ds[:], val_ds[:])
        mod2.to(device)
        loss = mod2.training_step(train_ds[:])['loss'].item()
        val_loss = mod2.training_step(val_ds[:])['loss'].item()
        return mod2, loss, val_loss

    def train_multistart(nlatent=1, nruns=10, include_offset=True, include_gain=True):

        #% fit 10 runs
        from copy import deepcopy
        mods = []
        losses = []
        val_losses = []

        for i in range(nruns):
            mod2, loss, val_loss = fit_gainmodel(mod1, nlatent=nlatent, include_offset=include_offset, include_gain=include_gain)
            mods.append(deepcopy(mod2))
            losses.append(loss)
            val_losses.append(val_loss)

        return mods, losses, val_losses

    def eval_model(mod2, ds, val_ds, use_average=False):

        use_average = False
        sample = ds[:]
        mod2 = mod2.to(device)
        rhat = mod2(sample).detach().cpu().numpy()

        if hasattr(mod2, 'latent_gain'):
            zg = mod2.latent_gain(sample['robs'])
            zg = mod2.gainnl(zg)
            
            if use_average:
                zg = mod2.readout_gain(zg).mean(dim=1).detach().cpu()
                zg = zg[:,None]
            else:
                zgav = mod2.readout_gain(zg).mean(dim=1).detach().cpu()
                zgav = zgav[:,None]
                # zgav = mod2.readout_gain(zg).mean(dim=1).detach().cpu()
                # zgav = zgav[:,None]
                zg = zg.detach().cpu()
        else:
            zgav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
            zg = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

        if hasattr(mod2, 'latent_offset'):
            zh = mod2.latent_offset(sample['robs'])
            if use_average:
                zh = mod2.readout_offset(zh).mean(dim=1).detach().cpu()
                zh = zh[:,None]
            else:
                zhav = mod2.readout_offset(zh).mean(dim=1).detach().cpu()
                zhav = zhav[:,None]
                zh = zh.detach().cpu()
        else:
            zhav = torch.zeros(sample['robs'].shape[0], 1, device='cpu')
            zh = torch.zeros(sample['robs'].shape[0], 1, device='cpu')

        sample = val_ds[:]
        robs_ = sample['robs'].detach().cpu()
        rhat_ = mod2(sample).detach().cpu()

        r2test = rsquared(robs_, rhat_)

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

    use_average = False
    moddict0 = eval_model(mod0, ds, val_ds, use_average=use_average)
    moddict0['model'] = mod0
    das['stim'] = moddict0

    moddict1 = eval_model(mod1, ds, val_ds, use_average=use_average)
    moddict1['model'] = mod1
    das['stimdrift'] = moddict1

    from copy import deepcopy
    ''' fit Affine model'''
    print("Fitting stim * gain + offset")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=10, include_offset=True, include_gain=True)
    bestid = np.nanargmin(vallosses)
    mod4 = deepcopy(models[bestid])
    moddict4 = eval_model(mod4, ds, val_ds, use_average=use_average)
    moddict4['model'] = mod4
    moddict4['valloss'] = np.asarray(vallosses)
    das['affine'] = moddict4
    print("Done")

    ''' fit offset model'''
    print("Fitting stim + offset")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=5, include_offset=True, include_gain=False)
    bestid = np.nanargmin(vallosses)
    mod2 = deepcopy(models[bestid])
    moddict2 = eval_model(mod2, ds, val_ds, use_average=use_average)
    moddict2['model'] = mod2
    moddict2['valloss'] = np.asarray(vallosses)
    das['offset'] = moddict2
    print("Done")

    ''' fit gain model'''
    print("Fitting stim * gain")
    models, losses, vallosses = train_multistart(nlatent=1, nruns=5, include_offset=False, include_gain=True)
    bestid = np.nanargmin(vallosses)
    mod3 = deepcopy(models[bestid])
    moddict3 = eval_model(mod3, ds, val_ds, use_average=use_average)
    moddict3['model'] = mod3
    moddict3['valloss'] = np.asarray(vallosses)
    das['gain'] = moddict3
    print("Done")

    # # pickle dataset
    import pickle

    with open(os.path.join(apath, aname), 'wb') as f:
        pickle.dump(das, f)

# if main
if __name__ == '__main__':

    fit_session(**vars)

