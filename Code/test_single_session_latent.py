#%% IMPORT
import os, sys

sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis

import matplotlib.pyplot as plt
# %matplotlib ipympl
plt.rcParams['pdf.fonttype'] = 42

fig_dir = '/mnt/Data/Figures/'

# Import torch
import torch
from torch import nn
from scipy.io import loadmat


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

from datasets import GenericDataset

%load_ext autoreload
%autoreload 2


#%% Load DATA

fpath = '/mnt/Data/Datasets/HuklabTreadmill/preprocessed_for_model/'

flist = os.listdir(fpath)

isess = 0

from fit_latents_session import get_data, get_dataloaders, eval_model
from models import SharedGain

ntents = 10
ds, dat = get_data(fpath, flist[isess], num_tents=ntents)

train_dl, val_dl = get_dataloaders(ds, batch_size=64, folds=5)


# %%
sample = ds[:]
NT, nstim = sample['stim'].shape
NC = sample['robs'].shape[1]

def fit_model(model, verbose=1):

    from torch.optim import AdamW
    from torch.optim.lr_scheduler import OneCycleLR
    from NDNT.training import Trainer, EarlyStopping

    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = OneCycleLR(optimizer, max_lr=0.01,
        steps_per_epoch=len(train_dl), epochs=20000)

    earlystopping = EarlyStopping(patience=500, verbose=False)

    trainer = Trainer(model, optimizer,
            scheduler,
            device=ds.device,
            optimize_graph=True,
            max_epochs=20000,
            early_stopping=earlystopping,
            log_activations=False,
            scheduler_after='batch',
            scheduler_metric=None,
            verbose=verbose)

    trainer.fit(model, train_dl, val_dl)


#%%

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

fit_model(mod0)

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

fit_model(mod1)

res0 = eval_model(mod0, ds, val_dl.dataset)
res1 = eval_model(mod1, ds, val_dl.dataset)

plt.figure()
plt.plot(res0['r2test'])
plt.plot(res1['r2test'])
plt.ylim([-0.1,1])
# %% fit affine model
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
mod2.drift.weight.requires_grad = False
mod2.stim.weight.requires_grad = True
mod2.bias.requires_grad = False
mod2.latent_gain.weight_scale = 1.0
mod2.latent_offset.weight_scale = 1.0
mod2.readout_gain.weight_scale = 1.0
mod2.readout_offset.weight_scale = 1.0

mod2.prepare_regularization()

fit_model(mod2)

#%%
res2 = eval_model(mod2, ds, val_dl.dataset)
# %%
plt.figure()
plt.plot(res0['r2test'], 'o')
plt.plot(res1['r2test'], 'o')
plt.plot(cids, res2['r2test'], 'o')
plt.ylim([-0.1,1])
# %%
plt.figure()
plt.plot(res2['zgain'].cpu(), res2['zoffset'].cpu(), '.')
plt.show()

# %%
plt.figure()
plt.plot(res1['r2test'][cids].cpu(), res2['r2test'].cpu(), 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
# %% Try fit session
from fit_latents_session import fit_session
apath = '/mnt/Data/Datasets/HuklabTreadmill/latent_modeling/'
aname = 'marmoset_23.pkl'
fname = aname.replace('.pkl', '.mat')

print(aname)

refit = True

if refit:
    a = fit_session(fpath, apath, fname, aname, ntents=5)

# %%
plt.figure()
plt.plot(a['stimdrift']['r2test'], a['affine']['r2test'], 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Stimulus + drift")
plt.ylabel("Affine")
# %%
