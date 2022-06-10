#%% IMPORT
from copy import deepcopy
import os, sys
import matplotlib

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2


#%% Load DATA
from fit_latents_session import fit_model, censored_lstsq, pca_train, cv_pca
fpath = '/mnt/Data/Datasets/HuklabTreadmill/preprocessed_for_model/'

flist = os.listdir(fpath)
flist = [f for f in flist if 'marmoset' in f]
# isess +=1# 52
isess = 15
# isess = 12
# isess = 6#48
# print(flist[isess])
fname = 'marmoset_%d.mat' %isess

isess = 18
fname = 'mouse_%d.mat' %isess

from fit_latents_session import get_data, get_dataloaders, eval_model
from models import SharedGain, SharedLatentGain

ntents = 5
ds, dat = get_data(fpath, fname, num_tents=ntents, normalize_robs=1)


robs = ds.covariates['robs']
dfs = ds.covariates['dfs']
robs = robs *dfs
robs = robs.detach().cpu().numpy()
plt.figure(figsize=(10,5))
plt.imshow(robs.T, aspect='auto', interpolation='none')
plt.colorbar()


robs = dat['robs']
plt.figure(figsize=(10,5))
# robs = robs * (robs < 10)
plt.imshow(robs.T > 12, aspect='auto', interpolation='none')
# plt.imshow(robs.T, aspect='auto', interpolation='none')
plt.colorbar()

# %matplotlib inline
plt.figure(figsize=(10,5))
r = (robs - robs.min(axis=0)) / (robs.max(axis=0) - robs.min(axis=0))
# r = r * (r < .6)
plt.imshow(r.T, aspect='auto', interpolation='none')
plt.colorbar()

plt.figure(figsize=(10,5))
r = (robs - robs.mean(axis=0)) / (robs.std(axis=0))
# r = r * (r < .6)
plt.imshow(r.T, aspect='auto', interpolation='none')
plt.colorbar()


#%%
%matplotlib ipympl
plt.figure(figsize=(10,5))
mu = np.mean(robs.astype('float32'), axis=0)
mad = np.median(np.abs(robs-mu), axis=0)
r = (robs - mu) / mad
# r = (robs - robs.mean(axis=0)) / (robs.std(axis=0))
# r =  r * (r < 12)
plt.imshow(r.T, aspect='auto', interpolation='none')
plt.colorbar()

plt.figure()
plt.hist(r.flatten(), bins=500)

#%%
plt.figure()
plt.plot(np.mean(robs, axis=0), np.var(robs, axis=0), '.')
plt.plot(np.mean(robs, axis=0), 10*np.mean(robs, axis=0))
plt.xlabel('mean')
plt.ylabel('var')

plt.figure()
f = plt.hist(np.var(robs, axis=0)/np.mean(robs, axis=0), bins=100)




#%%

sample = ds[:]
NT, nstim = sample['stim'].shape
NC = sample['robs'].shape[1]
print("%d Trials n % d Neurons" % (NT, NC))

# try to overfit data and throw out outliers
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
            reg_vals={'l2':0.01},
            act_func='lin')

from torch.utils.data import DataLoader
dl = DataLoader(ds, batch_size=64)
mod1.bias.requires_grad = False

t0 = fit_model(mod1, dl, dl, use_lbfgs=True, verbose=0)

mod1.to(device)
rhat = mod1(sample)
dfs = (rhat - sample['robs']).detach().cpu().abs() < 20
ds.covariates['dfs'] = torch.tensor(dfs.numpy(), dtype=torch.float32).to(device)
#%%
dfs = (rhat - sample['robs']).detach().cpu().abs()
plt.figure(figsize=(10,5))
plt.imshow(dfs.T, aspect='auto', interpolation='none')
plt.colorbar()

#%%
train_dl, val_dl, test_dl, indices = get_dataloaders(ds, batch_size=64, folds=4, use_dropout=True)

# U, data, rank, Mtrain = cv_pca(ds.covariates['robs'], rank=5, Mtrain=train_dl.dataset[:]['dfs']>0, Mtest=val_dl.dataset[:]['dfs']>0)
##%% MATRIX FACTORIZATION
train_err= []
test_err = []
data = ds.covariates['robs']
Mtrain = train_dl.dataset[:]['dfs']
Mtest = val_dl.dataset[:]['dfs']
maxrank = 25
terror = torch.zeros(maxrank, NC)
for rnk in range(1, maxrank):
    U, Vt, tre, te = cv_pca(ds.covariates['robs'], rank=rnk, Mtrain=train_dl.dataset[:]['dfs']>0, Mtest=val_dl.dataset[:]['dfs']>0)
    train_err.append((rnk, tre))
    test_err.append((rnk, te))
    resid = U@Vt - data
    mu = torch.sum(data*Mtrain, dim=0)/torch.sum(Mtrain, dim=0)

    total_err = data - mu

    tre = 1 - torch.sum(resid**2*Mtrain, dim=0) / torch.sum(total_err**2*Mtrain, dim=0)
    te = 1 - torch.sum(resid**2*Mtest, dim=0) / torch.sum(total_err**2*Mtest, dim=0)
    terror[rnk,:] = te.detach().cpu()


figs = []

fig, ax = plt.subplots(1, 1, figsize=(4, 3.5))
ax.plot(*list(zip(*train_err)), 'o-b', label='Train Data')
ax.plot(*list(zip(*test_err)), 'o-r', label='Test Data')
ax.set_ylabel('Var. Explained')
ax.set_xlabel('Number of PCs')
ax.set_title('PCA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.ylim(0, 1)
fig.tight_layout()
figs.append(fig)

'''
Step 0: check that the dataset has stable low-dimensional structure at >=4 dimensions

'''
rnk = 4
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

# Fit baseline and stimulus model
# ntents = 1

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
            stim_act_func='elu',
            stim_reg_vals={'l2':1},
            reg_vals={'l2':0.01},
            act_func='lin')

if ntents < 2:
    mod0.bias.requires_grad = True
else:
    mod0.bias.requires_grad = False

t1 = fit_model(mod0, train_dl, val_dl, use_lbfgs=True, verbose=0)


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
            stim_act_func='elu',
            stim_reg_vals={'l2':1},
            reg_vals={'l2':0.01},
            act_func='lin')

if ntents > 1:
    mod1.drift.weight.data = mod0.drift.weight.data.clone()
    mod1.bias.requires_grad = False
else:
    mod1.bias.data[:] = mod0.bias.data.clone()
    mod1.bias.requires_grad = True

t2 = fit_model(mod1, train_dl, val_dl, use_lbfgs=True, verbose=0)

res0 = eval_model(mod0, ds, val_dl.dataset)
res1 = eval_model(mod1, ds, val_dl.dataset)

cids = np.where(np.logical_and(res1['r2test'] > res0['r2test'], res1['r2test'] > 0))[0]

cids = np.intersect1d(cids, cids0)
cids = np.where(res1['r2test'] > 0)[0]
# cids = np.union1d(cids, cids0)
plt.figure()
plt.subplot(1,2,1)
plt.plot(res0['r2test'])
plt.plot(res1['r2test'])
plt.axhline(0, color='k', linestyle='--')
plt.ylim([-0.1,1])

plt.subplot(1,2,2)
plt.plot(res0['r2test'], res1['r2test'], 'o')
plt.plot(res0['r2test'][cids], res1['r2test'][cids], 'o', label='included in gain model')
plt.xlabel('Baseline model (Drift)')
plt.ylabel("Stimulus")
plt.plot(plt.xlim(), plt.xlim(), 'k--')
plt.xlim(-.5, 1)
plt.ylim(-.5, 1)

print("%d / %d units included in gain model" % (len(cids), NC))

# %% fit Autoencoder model first
from fit_latents_session import fit_latents, fit_gain_model, model_rsquared
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

""" Fit Offset Autoencoder"""
mod2 = SharedGain(nstim,
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
            stim_act_func='elu',
            stim_reg_vals={'l2': 1},
            reg_vals={'l2': .001},
            act_func='lin')


if ntents > 1:
    mod2.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
else:
    mod2.bias.requires_grad = True

mod2.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
mod2.bias.data = mod1.bias.data[cids].clone()
mod2.stim.weight.requires_grad = False
mod2.readout_offset.weight_scale = 1.0
mod2.latent_offset.weight_scale = 1.0
mod2.readout_offset.weight.data[:] = 1

mod2.prepare_regularization()

fit_model(mod2, train_dl, val_dl, use_lbfgs=True, verbose=0, use_warmup=True)

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

for i in range(1):
    """ Fit Affine Autoencoder"""
    mod3 = SharedGain(nstim,
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
                stim_act_func='elu',
                stim_reg_vals={'l2': 1},
                reg_vals={'l2': .1},
                act_func='lin')

    if ntents > 1:
        mod3.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
        mod3.drift.weight.requires_grad = True
        mod3.bias.requires_grad = False
    else:
        mod3.bias.requires_grad = True
    mod3.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
    mod3.bias.data = mod1.bias.data[cids].clone()
    mod3.stim.weight.requires_grad = False
    mod3.readout_gain.weight.data[:] = 1
    mod3.readout_offset.weight.data[:] = 1
    mod3.readout_offset.weight_scale = 1.0
    

    mod3.prepare_regularization()

    fit_model(mod3, train_dl, val_dl, use_lbfgs=True, verbose=0, use_warmup=True)

    res3 = eval_model(mod3, ds, val_dl.dataset)
    plt.plot(res3['zgain'])


mod3.to(device)
r2 = model_rsquared(mod3, val_dl.dataset[:])
mod3.readout_gain.weight.data[:,r2<0] = 0
mod3.readout_offset.weight.data[:,r2<0] = 0


res2 = eval_model(mod2, ds, test_dl.dataset)
res1 = eval_model(mod1, ds, test_dl.dataset)
res3 = eval_model(mod3, ds, test_dl.dataset)

plt.figure()
plt.subplot(1,2,1)
plt.plot(res1['r2test'][cids].cpu(), res2['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('stim')
plt.ylabel('offset')

plt.subplot(1,2,2)
plt.plot(res2['r2test'].cpu(), res3['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.ylabel('affine')
plt.xlabel('offset')

plt.figure()
plt.plot(res3['zgain'])
plt.plot(res3['zoffset'])



#%% shared gain matrix factorization

# train_dl, val_dl, test_dl, indices = get_dataloaders(ds, batch_size=264, folds=4, use_dropout=True)


mod3.to(device)
r2 = model_rsquared(mod3, val_dl.dataset[:])
# mod3.readout_gain.weight.data[:,r2<0] = 0
# mod3.readout_offset.weight.data[:,r2<0] = 0


mod4 = fit_gain_model(nstim, mod3, NC=NC, NT=len(ds),
    num_latent=1,
    cids=cids, ntents=ntents,
    train_dl=train_dl, val_dl=val_dl,
    include_gain=True,
    include_offset=True,
    verbose=0,
    l2s=[.1, 1],
    d2ts=[.001, .1, 1])

# mod4.to(device)
r24 = model_rsquared(mod4, val_dl.dataset[:])

# mod4.readout_gain.weight.data[:,r24<0] = 0
# mod4.readout_offset.weight.data[:,r24<0] = 0

# mod4.gain_mu.reg.vals

%matplotlib inline
mod3.training= False
mod4.training=False

plt.figure()
r2 = model_rsquared(mod1.to(device), val_dl.dataset[:])[mod4.cids]
r24 = model_rsquared(mod4.to(device), val_dl.dataset[:])
plt.plot(r2, r24, '.')
plt.plot((0,1), (0,1), 'k')
# plt.xlim((-.5, 1))
# plt.ylim((-.5, 1))
print('r2: %.4f, r2 lvm: %.4f' %(r2.mean().item(), r24.mean().item()))
# fit_gain_model(mod3)


res3 = eval_model(mod3, ds, test_dl.dataset)
res4 = eval_model(mod4, ds, test_dl.dataset)

plt.figure()
plt.subplot(1,2,1)
plt.plot(res1['r2test'][cids].cpu(), res3['r2test'].cpu(), '.')
plt.plot(res1['r2test'][cids].cpu(), res4['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('stim')
plt.ylabel('offset')

plt.subplot(1,2,2)
plt.plot(res3['r2test'].cpu(), res4['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('autoencoder')
plt.ylabel('latent')

#%%
%matplotlib ipympl
plt.figure()
plt.plot(res3['zgain']/res3['zgain'].var())
plt.plot(res4['zgain'], '-')


plt.figure()
plt.plot(res3['zoffset']/res3['zoffset'].var())
plt.plot(res4['zoffset'])

plt.figure()
plt.plot(dat['runningspeed'])

#%%

mod4 = SharedLatentGain(nstim,
                NC=NC,
                NT=len(ds),
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                include_stim=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2': 1},
                reg_vals={'d2t': .01, 'l2': 0.001},
                readout_reg_vals={'l1': 0.001, 'l2': 0.001})

fixweights = False
loss, model = fit_latents(mod4, mod3, train_dl, val_dl, fit_sigmas=True, max_iter=10,
    seed=None, fix_readout_weights=fixweights)


# model.gain_mu.weight.data[:] = res3['zgain'].clone()
# model.offset_mu.weight.data[:] = res3['zoffset'].clone()
model.training = False

res3 = eval_model(mod3, ds, test_dl.dataset)
res4 = eval_model(model, ds, test_dl.dataset)

plt.figure()
plt.subplot(1,2,1)
plt.plot(res1['r2test'][cids].cpu(), res3['r2test'].cpu(), '.')
plt.plot(res1['r2test'][cids].cpu(), res4['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('stim')
plt.ylabel('offset')

plt.subplot(1,2,2)
plt.plot(res3['r2test'].cpu(), res4['r2test'].cpu(), '.')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel('autoencoder')
plt.ylabel('latent')

plt.figure()
plt.plot(res3['zgain'])
plt.plot(res4['zgain'], '--')


plt.figure()
plt.plot(res3['zoffset'])
plt.plot(res4['zoffset'])
#%% fit latent model
%matplotlib ipympl
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

replicates = 2
losses = []
models = []
gain_mu = []
offset_mu = []

# d2xs = [0, 0.001, 0.01, 0.1, .5, 1, 5, 10, 50]
d2xs = [1, 10, 50, 100, 500, 1000]
replicates = len(d2xs)
plt.figure(figsize=(10,5))

for r in range(replicates):
    mod3 = SharedLatentGain(nstim,
                NC=NC,
                NT=len(ds),
                cids=cids,
                num_latent=1,
                num_tents=ntents,
                include_stim=True,
                include_gain=True,
                include_offset=True,
                tents_as_input=False,
                output_nonlinearity='Identity',
                stim_act_func='lin',
                stim_reg_vals={'l2':1},
                reg_vals={'d2t': d2xs[r], 'l2': 0.001},
                readout_reg_vals={'l1': 0.001, 'l2':0.001})

    fixweights = False#r < 1
    loss, model = fit_latents(mod3, mod1, train_dl, fit_sigmas=True, max_iter=10,
        seed=None, fix_readout_weights=fixweights)
    
    res = eval_model(model, ds, val_dl.dataset)
    model0 = deepcopy(model)
    r2 = res['r2test'].mean().numpy()
    print('l1: %.3f, r2=%.3f' %(model.readout_gain.reg.vals['l1'], r2))

    # increase 
    
    while True:
        model.readout_gain.reg.set_reg_val('l1', model.readout_gain.reg.vals['l1']*10)
        loss, model = fit_latents(model, None, train_dl, fit_sigmas=False, max_iter=10,
            seed=None, fix_readout_weights=fixweights)
        res = eval_model(model, ds, val_dl.dataset)
        print('l1: %.3f, r2=%.3f' %(model.readout_gain.reg.vals['l1'], res['r2test'].mean().numpy()))

        if res['r2test'].mean().numpy() < r2:
            print('tolerance reached')
            break
        else:
            model0 = deepcopy(model)
            r2 = res['r2test'].mean().numpy()
    
    
    losses.append(loss)
    models.append(model0)
    print('Fit run %d: %.3f' % (r, loss))


    plt.plot(mod3.gain_mu.weight.detach().cpu())
    gain_mu.append(mod3.gain_mu.weight.detach().cpu().clone())
    offset_mu.append(mod3.offset_mu.weight.detach().cpu().clone())

id = np.argmin(np.asarray(losses))
mod3 = models[id]

res3 = eval_model(mod3, ds, val_dl.dataset)
#%%
plt.figure()
plt.subplot(1,3,1)
plt.plot(res1['r2test'][cids].cpu(), res2['r2test'].cpu(), 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Stim Model")
plt.ylabel("Autoencoder Latent Model")

ax1 = plt.subplot(1,3,2)
ax2 = plt.subplot(1,3,3)
for id in range(len(models)):
    res3 = eval_model(models[id], ds, test_dl.dataset)
    
    ax1.plot(res1['r2test'][cids].cpu(), res3['r2test'].cpu(), '.')

    
    ax2.plot(res2['r2test'].cpu(), res3['r2test'].cpu(), '.')

ax1.plot((0,1), (0,1), 'k')
ax1.set_xlabel("Stim Model")
ax1.set_ylabel("Fit Latent Model")

ax2.plot((0,1), (0,1), 'k')
ax2.set_xlabel("Autoencoder Model")
ax2.set_ylabel("Fit Latent Model")

#%%
plt.figure()
r2test = []
for id in range(replicates):
    res3 = eval_model(models[id], ds, test_dl.dataset)
    r2test.append(res3['r2test'].numpy())
    # plt.plot(d2xs[id], res3['r2test'].mean().cpu(), '.')



r2 = np.row_stack(r2test)

plt.figure()
plt.plot(np.mean(r2,axis=1))

plt.figure()
f = plt.plot(r2 - r2[0,:])


#%%
%matplotlib inline
x = np.concatenate(offset_mu, axis=1)
plt.figure(figsize=(10,10))
for i in range(replicates):
    for j in range(replicates):
        plt.subplot(replicates, replicates, i*replicates + j + 1)
        plt.plot(x[:,i], x[:,j], '.')

x = np.concatenate(gain_mu, axis=1)
plt.figure(figsize=(10,10))
for i in range(replicates):
    for j in range(replicates):
        plt.subplot(replicates, replicates, i*replicates + j + 1)
        plt.plot(x[:,i], x[:,j], '.')        

#%%
%matplotlib ipympl
plt.figure(figsize=(10,4))
ax = plt.subplot(1,1,1)
ax.imshow(np.sqrt(robs.T.detach().cpu().numpy()), aspect='auto', cmap='coolwarm', interpolation='none')
ax2 = ax.twinx()

# plt.figure(figsize=(10,4))
for id in range(replicates):
    res3 = eval_model(models[id], ds, train_dl.dataset)
    if hasattr(models[id], 'gain_mu'):
        plt.plot(models[id].gain_mu.weight.detach().cpu(), 'r')
    
    if hasattr(models[id], 'offset_mu'):
        plt.plot(models[id].offset_mu.weight.detach().cpu(), 'g')
    plt.xlim(0, robs.shape[0])
    # plt.plot(res3['zgain'])
plt.show()


#%%
%matplotlib ipympl
plt.figure(figsize=(10,5))
ax = plt.subplot()
plt.plot(res2['zgain'])
plt.plot(res3['zgain'])
ax2 = ax.twinx()
plt.plot(ds.covariates['runningspeed'].cpu().numpy(), 'k')

#%%
plt.figure()
plt.plot(mod3.readout_gain.weight.detach().cpu().T)

#%%
plt.figure()
plt.plot(res3['zgain'].cpu(), ds.covariates['runningspeed'].cpu(), '.')
plt.show()

isrunning = ds.covariates['runningspeed'].cpu() > 3
plt.figure()
bins = np.linspace(np.min(res3['zgain'].cpu().numpy()), np.max(res3['zgain'].cpu().numpy()), 50)
h1 = plt.hist(res3['zgain'].cpu()[isrunning].T.numpy(), alpha=.5, bins=bins, density=True)
h2 = plt.hist(res3['zgain'].cpu()[~isrunning].T.numpy(), alpha=.5, bins=bins, density=True)



#%%
# # %%
# plt.figure()
# plt.plot(res0['r2test'], 'o')
# plt.plot(res1['r2test'], 'o')
# plt.plot(cids, res2['r2test'], 'o')
# plt.ylim([-0.1,1])

##%% redo PCA on fit neurons
train_err= []
test_err = []
for rnk in range(1, 25):
    U, Vt, tre, te = cv_pca(ds.covariates['robs'][:,cids], rank=rnk, Mtrain=train_dl.dataset[:]['dfs'][:,cids]>0, Mtest=val_dl.dataset[:]['dfs'][:,cids]>0)
    train_err.append((rnk, tre))
    test_err.append((rnk, te))
## %% Compare PCA to latent model
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
ax.plot(*list(zip(*train_err)), 'o-b', label='PCA Train Data')
ax.plot(*list(zip(*test_err)), 'o-r', label='PCA Test Data')
ax.plot(1, res1['r2test'].mean(), 'o', label='Stim Model', color='k')
ax.plot(1, res2['r2test'].mean(), 'o', label='1 Gain Latent Model', color='m')
ax.axhline(res3['r2test'].mean(),color='m')
# ax.plot(1, res2['r2test'].mean(), 'o', label='1 Gain Autoencoder Model', color='g')
ax.set_ylabel('Var. Explained')
ax.set_xlabel('Number of PCs')
ax.set_title('PCA vs. 1D Shared Gain')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.ylim(0, 1)
fig.tight_layout()

#%% plot how stimulus weights changed
w1 = mod1.stim.get_weights()[:,cids]
w2 = mod3.stim.get_weights()

N = w1.shape[1]
sx = np.ceil(np.sqrt(N)).astype(int)
sy = np.round(np.sqrt(N)).astype(int)
for cc in range(N):
    plt.subplot(sx, sy, cc + 1)
    plt.plot(w1[:,cc], 'b')
    plt.plot(w2[:,cc], 'r')

