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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32

NUM_WORKERS = int(os.cpu_count() / 2)

%load_ext autoreload
%autoreload 2


#%% Load DATA

fpath = '/mnt/Data/Datasets/HuklabTreadmill/preprocessed_for_model/'

flist = os.listdir(fpath)
for i in range(len(flist)):
    print('%d) %s' % (i, flist[i]))


#%% 
def fit_model(model, train_dl, val_dl,
    lr=1e-3, max_epochs=5,
    wd=0.01,
    max_iter=10000,
    use_lbfgs=False, verbose=0, early_stopping_patience=10,
    use_warmup=True, seed=1234):

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

            trainer.fit(model, train_dl, val_dl)


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


def cv_pca(data, rank, Mtrain=None, Mtest=None, p_holdout=0.2):
    """Fit PCA or NMF while holding out a fraction of the dataset.
    """

    # choose solver for alternating minimization
    solver = censored_lstsq

    # create masking matrix
    if Mtrain is None:
        Mtrain = torch.rand(*data.shape, device=data.device) > p_holdout
    
    if Mtest is None:
        Mtest = ~Mtrain

    Mtrain = Mtrain.to(data.device)
    Mtest = Mtest.to(data.device)

    # initialize U randomly
    U = torch.randn(data.shape[0], rank, device=data.device)

    # return U, data, rank, Mtrain
    Vt = solver(U, data, Mtrain)
    resid = U@Vt - data
    mse0 = torch.mean(resid**2)
    tol = 1e-3
    # fit pca/nmf
    for itr in range(10):
        Vt = solver(U, data, Mtrain)
        U = solver(Vt.T, data.T, Mtrain.T).T
        resid = U@Vt - data
        mse = torch.mean(resid[Mtrain]**2)
        # print('%d) %.3f' %(itr, mse))
        if mse > (mse0 - tol):
            break
        mse0 = mse

    # return result and test/train error
    resid = U@Vt - data
    total_err = data - torch.mean(data, dim=0)
    train_err = 1 - torch.sum(resid[Mtrain]**2) / torch.sum(total_err[Mtrain]**2)
    test_err = 1 - torch.sum(resid[Mtest]**2) / torch.sum(total_err[Mtest]**2)
    return U, Vt, train_err, test_err








#%% LOAD DATA
# isess +=1# 52
isess = 6
print(flist[isess])
from fit_latents_session import get_data, get_dataloaders, eval_model
from models import SharedGain, SharedLatentGain

ntents = 10
ds, dat = get_data(fpath, flist[isess], num_tents=ntents, normalize_robs=True, zthresh=5)

train_dl, val_dl, indices = get_dataloaders(ds, batch_size=64, folds=5, use_dropout=True)

sample = ds[:]
NT, nstim = sample['stim'].shape
NC = sample['robs'].shape[1]
# U, data, rank, Mtrain = cv_pca(ds.covariates['robs'], rank=5, Mtrain=train_dl.dataset[:]['dfs']>0, Mtest=val_dl.dataset[:]['dfs']>0)
##%% MATRIX FACTORIZATION
train_err= []
test_err = []
for rnk in range(1, 25):
    U, Vt, tre, te = cv_pca(ds.covariates['robs'], rank=rnk, Mtrain=train_dl.dataset[:]['dfs']>0, Mtest=val_dl.dataset[:]['dfs']>0)
    train_err.append((rnk, tre))
    test_err.append((rnk, te))

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


cids = np.arange(NC)
rnk = 6
_, sid = np.where(ds.covariates['stim'].cpu().numpy())
inds = np.argsort(sid)
sortit = False
dfs = ds[:]['dfs'][:,cids]>0

robs = ds.covariates['robs'][:,cids]*dfs
if sortit:
    robs = robs[inds,:]


U, Vt, tre, te = cv_pca(robs, rank=rnk, Mtrain=train_dl.dataset[:]['dfs'][:,cids]>0, Mtest=val_dl.dataset[:]['dfs'][:,cids]>0)
plt.figure(figsize=(8, 4.5))
plt.imshow(robs.cpu().numpy().T, aspect='auto', interpolation='none')

plt.figure(figsize=(8, 4.5))
plt.imshow( (U@Vt).T.cpu().numpy(), aspect='auto', interpolation='none')


plt.figure(figsize=(8, 4.5))
rm = robs.cpu().numpy()*dfs.cpu().numpy()

plt.imshow(robs.cpu().numpy().T - rm.T, aspect='auto', interpolation='none')

#%%
# filter the data using scipy convolve 2
from scipy.ndimage import uniform_filter
robs_smooth = uniform_filter(robs.cpu().numpy(), size=(10, 1), mode='constant')
x = robs.cpu().numpy().T / robs_smooth.T
plt.imshow(x, aspect='auto', interpolation='none')
plt.colorbar()

# plt.figure(figsize=(8, 4.5))
# plt.plot(x.T)



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
            latent_noise=True,
            stim_act_func='lin',
            stim_reg_vals={'l2':0.0},
            reg_vals={'l2':0.0},
            act_func='lin')

fit_model(mod0, train_dl, val_dl, use_lbfgs=True, verbose=2)


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

fit_model(mod1, train_dl, val_dl, use_lbfgs=True, verbose=2)

res0 = eval_model(mod0, ds, val_dl.dataset)
res1 = eval_model(mod1, ds, val_dl.dataset)

plt.figure()
plt.plot(res0['r2test'])
plt.plot(res1['r2test'])
plt.ylim([-0.1,1])
# %% fit affine model
cids = np.where(np.logical_and(res1['r2test'] > res0['r2test'], res1['r2test'] > 0))[0]
cids = np.where(res1['r2test'] > 0)[0]

seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

mod2 = SharedGain(nstim,
            NC=NC,
            cids=cids,
            num_latent=1,
            num_tents=ntents,
            latent_noise=True,
            include_stim=True,
            include_gain=True,
            include_offset=False,
            tents_as_input=False,
            output_nonlinearity='Identity',
            stim_act_func='lin',
            stim_reg_vals={'l2':0.01},
            reg_vals={'l2': 1},
            act_func='lin')


mod2.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
mod2.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
mod2.bias.data = mod1.bias.data[cids].clone()

mod2.stim.weight.requires_grad = True

if hasattr(mod2, 'readout_offset'):
    mod2.readout_offset.weight_scale = 1.0
    mod2.latent_offset.weight_scale = 1.0
    mod2.drift.weight.requires_grad = False
    mod2.bias.requires_grad = False

mod2.prepare_regularization()

fit_model(mod2, train_dl, val_dl, use_lbfgs=True, verbose=2, use_warmup=True)

res2 = eval_model(mod2, ds, val_dl.dataset)
plt.figure()
plt.plot(res1['r2test'][cids].cpu(), res2['r2test'].cpu(), 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')

plt.figure()
plt.plot(res2['zgain'])

#%% shared gain matrix factorization
mod3 = SharedLatentGain(nstim,
            NC=NC,
            NT=len(ds),
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
            reg_vals={'d2t': .1, 'l2': 0.001},
            readout_reg_vals={'l2': 1})


mod3.logvar_g.data[:] = 1
mod3.logvar_g.requires_grad = False
# mod3.gain_mu.weight.data[:] = res2['zgain']
# mod3.gain_mu.weight.data[:] *= 10

# mod3.gain_mu.weight_scale = 1.0
# mod3.readout_gain.weight_scale = 1.0
# mod3.readout_gain.weight.data[:] = 1

mod3.drift.weight.data = mod1.drift.weight.data[:,cids].clone()
mod3.stim.weight.data = mod1.stim.weight.data[:,cids].clone()
mod3.bias.data = mod1.bias.data[cids].clone()

mod3.stim.weight.requires_grad = True
mod3.drift.weight.requires_grad = True
mod3.bias.requires_grad = True
# fit_model(mod3, train_dl, val_dl, use_lbfgs=False, verbose=1, use_warmup=True, lr=.001, wd=0.01, max_epochs=20000, early_stopping_patience=100)

# fit_model(mod3, train_dl, val_dl, use_lbfgs=True, verbose=2, use_warmup=True, max_iter=1000)

# #%%
for itr in range(10):
    mod3.gain_mu.weight.requires_grad = True
    mod3.readout_gain.weight.requires_grad = False
    fit_model(mod3, train_dl, val_dl, use_lbfgs=True, verbose=2)

    mod3.gain_mu.weight.requires_grad = False
    mod3.readout_gain.weight.requires_grad = True
    fit_model(mod3, train_dl, val_dl, use_lbfgs=True, verbose=2)

# mod3.gain_mu.weight.requires_grad = True

# fit_model(mod3, train_dl, val_dl, use_lbfgs=True, verbose=2, use_warmup=True, lr=.0001, wd=0.0)
# plt.plot(mod3.gain_mu.weight.detach().cpu())
#%%
# res2 = eval_model(mod2, ds, val_dl.dataset)
res3 = eval_model(mod3, ds, train_dl.dataset)

plt.figure()
ax = plt.subplot()
plt.plot(res2['zgain'])
plt.plot(res3['zgain'])
ax2 = ax.twinx()
plt.plot(ds.covariates['runningspeed'].cpu().numpy(), 'k')

plt.figure()
plt.plot(mod3.readout_gain.weight.detach().cpu().T)
plt.plot(mod2.readout_gain.weight.detach().cpu().T)

plt.figure()
plt.plot(res2['zgain'].cpu(), res3['zgain'].cpu(), '.')
plt.show()

plt.figure()
plt.subplot(1,2,1)
plt.plot(res1['r2test'][cids].cpu(), res2['r2test'].cpu(), 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Stim Model")
plt.ylabel("Autoencoder Latent Model")
plt.subplot(1,2,2)
plt.plot(res1['r2test'][cids].cpu(), res3['r2test'].cpu(), 'o')
plt.plot(plt.xlim(), plt.xlim(), 'k')
plt.xlabel("Stim Model")
plt.ylabel("Fit Latent Model")
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
ax.plot(1, res3['r2test'].mean(), 'o', label='1 Gain Latent Model', color='m')
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

#%%
cids = np.arange(NC)
rnk = 1
robs = ds.covariates['robs'][:,cids]
dfs = ds[:]['dfs'][:,cids]>0
U, Vt, tre, te = cv_pca(robs, rank=rnk, Mtrain=train_dl.dataset[:]['dfs'][:,cids]>0, Mtest=val_dl.dataset[:]['dfs'][:,cids]>0)
plt.figure(figsize=(8, 4.5))
plt.imshow(ds.covariates['robs'][:,cids].cpu().numpy().T, aspect='auto', interpolation='none')

plt.figure(figsize=(8, 4.5))
plt.imshow( (U@Vt).T.cpu().numpy(), aspect='auto', interpolation='none')


plt.figure(figsize=(8, 4.5))
rm = robs.cpu().numpy()*dfs.cpu().numpy()

plt.imshow(robs.cpu().numpy().T - rm.T, aspect='auto', interpolation='none')


# %%
##%% redo PCA on noise
X = torch.randn(NT, NC)
# X = robs
train_err= []
test_err = []
for rnk in range(1, 25):
    U, Vt, tre, te = cv_pca(X, rank=rnk)
    train_err.append((rnk, tre))
    test_err.append((rnk, te))
## %% Compare PCA to latent model
fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
ax.plot(*list(zip(*train_err)), 'o-b', label='PCA Train Data')
ax.plot(*list(zip(*test_err)), 'o-r', label='PCA Test Data')
# ax.plot(1, res1['r2test'].mean(), 'o', label='Stim Model', color='k')
# ax.plot(1, res3['r2test'].mean(), 'o', label='1 Gain Latent Model', color='m')
# ax.axhline(res3['r2test'].mean(),color='m')
# ax.plot(1, res2['r2test'].mean(), 'o', label='1 Gain Autoencoder Model', color='g')
ax.set_ylabel('Var. Explained')
ax.set_xlabel('Number of PCs')
ax.set_title('PCA on noise')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend()
plt.ylim(-1, 1)
fig.tight_layout()