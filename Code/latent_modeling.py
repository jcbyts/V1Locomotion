#%% Import
import os, sys
from turtle import color

sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np
# from sklearn.decomposition import FactorAnalysis

import matplotlib.pyplot as plt
import seaborn as sns
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
from scipy.stats import wilcoxon, mannwhitneyu

%load_ext autoreload
%autoreload 2


#%% Load DATA
overwrite = False
trial_thresh = 250
neuron_thresh = 10
running_frac_thresh = (0.05, .95)

import pickle
from fit_latents_session import fit_session
fpath = '/mnt/Data/Datasets/HuklabTreadmill/preprocessed_for_model/'
fix_readout_weights = False
if fix_readout_weights:
    apath = '/mnt/Data/Datasets/HuklabTreadmill/latent_modeling_fixed/'
else:
    apath = '/mnt/Data/Datasets/HuklabTreadmill/latent_modeling/'

from NDNT.utils import ensure_dir
ensure_dir(apath)

flist = os.listdir(fpath)

# subject = 'mouse'
# flist = [f for f in flist if subject in f]
nsessions = len(flist)
# nsessions = 10
sess_success = np.zeros(nsessions)
sessinfo = []

for isess in range(0, nsessions):
    print(isess)
    fname = flist[isess]
    aname = fname.replace('.mat', '.pkl')

    # try:
    # check if file exists
    if overwrite or not os.path.isfile(apath + aname):
        print("fitting model")
        fit_session(fpath, apath, fname, aname)
        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)
    else:
        print("Loading analyses")
        import pickle
        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)
    sess_success[isess] = 1
    subject =  aname[:aname.find('_')]
    id = int(aname[aname.find('_')+1:aname.find('.')])
    NT, NC = das['data']['robs'].shape
    nCids = len(das['affine']['model'].cids)
    sessinfo.append({'subj': subject, 'id': id, 'NT': NT, 'NC': NC, 'nCids': nCids})
    # except:
    #     print("Failed to fit model")
    #     continue


#%% Plot summary about number of neurons
subjs = ['marmoset', 'mouse']

NT = []
NC = []
nCids = []
for subj in subjs:
    sessinds = np.where([s['subj'] == subj for s in sessinfo])[0]
    print("Found %d sessions for subject %s" % (len(sessinds), subj))
    
    NT.append(np.asarray([sessinfo[i]['NT'] for i in sessinds]))
    NC.append(np.asarray([sessinfo[i]['NC'] for i in sessinds]))
    nCids.append(np.asarray([sessinfo[i]['nCids'] for i in sessinds]))

    for isess in range(len(sessinds)):
        sess = sessinfo[sessinds[isess]]
        print("%d) %s %d, %d trials, %d units (%d good)" % (isess, sess['subj'], sess['id'], sess['NT'], sess['NC'], sess['nCids']))

figures = []

figures.append(plt.figure(figsize=(10,4)))

plt.subplot(1,2,1)
for isubj, subj in enumerate(subjs):
    plt.plot(NC[isubj], nCids[isubj], 'o', label=subj)
plt.plot(plt.xlim(), plt.xlim(), '--', color='black')
plt.xlabel("# Neurons Total")
plt.ylabel("# Neurons Good")
plt.xlim(0, 100)
plt.ylim(0, 100)


plt.subplot(1,2,2)
for isubj, subj in enumerate(subjs):
    plt.plot(NT[isubj], nCids[isubj], 'o', label=subj)
plt.xlabel("# Trials")
plt.ylabel("# Neurons Good")
plt.xlim(0, 1200)


#%% plot R observed and Gain / Offset + model comparison
import matplotlib.gridspec as gridspec
import matplotlib.backends.backend_pdf
import matplotlib
%matplotlib inline
plt.close('all')
figures = []

def skip_session(das, neuron_thresh=10, trial_thresh=300, run_frac=(.10, .90)):
        
        msg = ''

        if len(das['affine']['model'].cids) < neuron_thresh:
            msg = msg + " Skipping session %s because too few neurons" %aname

        # check fraction running
        fracrun = np.mean(das['data']['running'] > 3)
        if fracrun > run_frac[1] or fracrun < run_frac[0]:
            msg = msg + " Skipping session %s because fraction running is extreme" %aname
            
        if len(das['data']['running']) < trial_thresh:
            msg = msg + 'Skipping session %s because # trials too low' %aname

        
        skip = not msg==''
        return skip, msg


def plot_summary(das, aname, modelname='affine'):
    gridspec.GridSpec(2,3)

    model = das[modelname]['model']
    zgain = model.gain_mu.get_weights()
    zweight = model.readout_gain.get_weights()
    if np.mean(zweight) < 0: # flip sign if both are negative
        zgain *= -1
        zweight *= -1
    
    zoffset = model.offset_mu.get_weights()
    zoffweight = model.readout_offset.get_weights()
    if np.mean(zoffweight) < 0: # flip sign if both are negative
        zoffset *= -1
        zoffweight *= -1

    dfs = das['data']['dfs'][:,das[modelname]['model'].cids]
    robs = das['data']['robs'][:,das[modelname]['model'].cids]*dfs
    # robs = robs

    ind = np.argsort(zweight)
    ax0 = plt.subplot2grid((6,3), (0,0), colspan=2, rowspan=3)

    plt.imshow(robs[:,ind].T, aspect='auto', interpolation='none', cmap='jet')
    # plt.title(aname.replace('.pkl', ''))
    ax0.set_xticklabels([])
    

    ax = plt.subplot2grid((6,3), (3,0), colspan=2)        
    plt.plot(das['data']['running'], 'k', label='Running')
    ax.set_xlim((0, robs.shape[0]))
    ax.set_ylabel('running')
    # ax.legend()
    # ax = plt.subplot(2,1,2)

    # ax2 = ax.twinx()
    ax2 = plt.subplot2grid((6,3), (4,0), colspan=2)
    plt.plot(zgain, 'r', label='Gain')
    ax2.set_xlim((0, robs.shape[0]))
    # ax2.legend()
    ax2.set_ylabel('Gain')

    ax3 = plt.subplot2grid((6,3), (5,0), colspan=2)
    plt.plot(zoffset, 'b', label='Offset')
    ax3.set_xlim((0, robs.shape[0]))
    # ax3.legend()
    ax3.set_ylabel('Offset')

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
    ax0.set_title(aname.replace('.pkl', '') + '  ' + titlestr)

    plt.subplot2grid((2,3), (0,2), colspan=1)
    plt.plot(das['stimdrift']['r2test'], das[modelname]['r2test'], 'o')
    plt.plot((0,1), (0,1), 'k')
    plt.xlabel('stim')
    plt.ylabel(modelname)

    plt.subplot2grid((2,3), (1,2), colspan=1)
    x = das[modelname]['r2test']-das['stimdrift']['r2test']
    x = x.numpy()
    plt.hist(x)
    plt.plot(np.median(x), plt.ylim()[1], 'v')
    plt.xlabel('affine - stim')

#%%

for subject in ['mouse', 'marmoset']:

    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)

        skip, msg = skip_session(das)

        if skip:
            print(msg)
            continue

        figures.append(plt.figure(figsize=(10,5)))
        plot_summary(das, aname)
        

pdf = matplotlib.backends.backend_pdf.PdfPages("robs_and_gain.pdf")
for fig in figures: ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()

#%% plot model comparison summary
%matplotlib inline

clr_mouse = np.asarray([206, 110, 41])/255
clr_marmoset = np.asarray([51, 121, 169])/255

fig = plt.figure(figsize=(4,4))
ax = plt.subplot()

fig2 = plt.figure()
ax2 = plt.subplot()

for subject in ['mouse', 'marmoset']:

    
    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)


        if subject=='mouse':
            clr = clr_mouse
        else:
            clr = clr_marmoset

        skip, msg = skip_session(das, trial_thresh=trial_thresh, run_frac=running_frac_thresh)

        if skip:
            print(msg)
            continue

        r2stim = das['stimdrift']['r2test'].mean().cpu()
        r2affine = das['affine']['r2test'].mean().cpu()
        ax.plot(r2stim, r2affine, '.', color=clr)
        
        nCids = len(das['affine']['model'].cids)
        # mdiff = r2affine - das['offset']['r2test'].mean().cpu()
        # ax2.plot(nCids, mdiff, '.', color=clr)

        NT = das['data']['robs'].shape[0]
        print("%s: NT=%d NC=%d r2stim = %.4f r2affine = %.4f" %(aname, NT, nCids, r2stim, r2affine))
        mdiff = r2affine - r2stim
        ax2.plot(NT, mdiff, '.', color=clr)

ax.plot((0,.5),(0,.5), 'k')
ax.set_xlabel('Stim model')
ax.set_ylabel("Affine model")
ax.set_xlim( ( 0, .5))
sns.despine(ax=ax, trim=True)
fig.savefig("../Figures/model_compare_session.pdf")

ax2.axhline(0, color='k')
# ax2.set_xlabel("# neurons")
ax2.set_xlabel("# trials")
ax2.set_ylabel('Model improvement with gain')
sns.despine(trim=True)
#%%
from scipy.stats import spearmanr
import pickle

def dprime(x, ix):

    mu1 = x[ix.flatten(),:].mean(dim=0)
    mu2 = x[~ix.flatten(),:].mean(dim=0)
    sd = x.std(dim=0)
    dp = (mu1 - mu2) / sd
    return dp.mode()[0].item()

def sessstats(das, fname=None, modelname='affine'):
    
    import torch.nn.functional as F

    r2affine = das[modelname]['r2test']
    r2stim = das['stimdrift']['r2test']
    r2offset = das['offset']['r2test']


    running = das['data']['running']
    pupil = das['data']['pupil']
    mod = das[modelname]['model']
    mod.to('cpu')

    robs = torch.tensor(das['data']['robs'])

    # latent gain gets datafiltered input
    s = robs.std(dim=0)
    mu = robs.mean(dim=0)
    dfs = das['data']['dfs']

    robs = robs * dfs

    zg = das[modelname]['zgain'] #mod.latent_gain(robs)
    zh = das[modelname]['zoffset']

    # get the average sign of the gain to find positive
    sflipg = np.sign(np.median(mod.readout_gain.get_weights()))
    sfliph = np.sign(np.median(mod.readout_offset.get_weights()))

    zglatent = F.relu(1 + zg*sflipg).detach().cpu() # convert latent to a single gain
    zhlatent = zh.detach().cpu() * sfliph # flip sign if most of the population has a negative weight

    # get population level gains
    zgpop = F.relu(1 + mod.readout_gain(zg).detach().cpu())
    # get population level offsets
    zhpop = mod.readout_offset(zh).detach().cpu()

    zglatent[zglatent < 1e-6] = 1e-6
    # zg = (zglatent).log2()

    zg *= sflipg
    zh *= sfliph

    # get gain dprime running
    dprun = dprime(zg, running>3)
    dppup = dprime(zg, pupil > np.nanmedian(pupil))

    res = spearmanr(running, pupil, nan_policy='omit')

    dstat = {}
    dstat['sess'] = '%s %i' %(subject, isess)
    dstat['fname'] = fname
    dstat['gainrange'] = zgpop.std(dim=0).numpy()
    dstat['offrange'] = zhpop.std(dim=0).numpy()
    dstat['dprungain'] = dprun
    dstat['dppupgain'] = dppup
    dstat['runningpupilcorr'] = res
    dstat['gainruncorr'] = spearmanr(zglatent, running, nan_policy='omit')
    dstat['gainpupilcorr'] = spearmanr(zglatent, pupil, nan_policy='omit')
    dstat['offsetruncorr'] = spearmanr(zhlatent, running, nan_policy='omit')
    dstat['offsetpupilcorr'] = spearmanr(zhlatent, pupil, nan_policy='omit')
    dstat['running'] = running
    dstat['pupil'] = pupil
    dstat['zglatent'] = zglatent.numpy()
    dstat['zhlatent'] = zhlatent.numpy()
    dstat['zg'] = zg.numpy()
    dstat['zh'] = zh.numpy()

    modellist = ['drift', 'stimdrift', 'offset', 'gain', 'affine', 'affineae']
    dstat['r2models'] = {}
    for f in modellist:
        dstat['r2models'][f] = das[f]['r2test'].numpy()

    return dstat

#%%
dstats = []
subjid = []

thelist = [fname.replace('.mat', '.pkl') for fname in flist[0:6]]

for subject in ['mouse', 'marmoset']:

    
    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        subjid.append(subject)
        aname = filelist[isess]

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)
        
        

        dstats.append(sessstats(das, aname, modelname='affine'))


gainrange = np.concatenate([d['gainrange'] for d in dstats])
subjid = np.concatenate([ np.ones(len(d['gainrange']))*('mouse' in d['sess']) for d in dstats])
NCs = np.concatenate([len(d['r2models']['gain'])*[len(d['r2models']['gain'])] for d in dstats])
NTs = np.concatenate([len(d['r2models']['gain'])*[len(d['running'])] for d in dstats])


#% Model performance
def fancy_violinplot(r2plot, modellist, clr, bw='scott', inner='quartile'):
    import seaborn as sns
    from scipy.stats import mannwhitneyu
    
    # sns.violinplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]), y=np.concatenate(r2plot),
        # color=clr, bw=bw, inner=inner, cut=0)
    

    sns.stripplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]),
        y=np.concatenate(r2plot),
        color=clr, size=2, dodge=True, jitter=.35, alpha=.25, zorder=1)

    sns.pointplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]), y=np.concatenate(r2plot),
        color='black', estimator=np.median, markers='o', scale=1, dodge=True)

    

    n = len(r2plot)
    import itertools
    for i in itertools.combinations(range(n),2):
        x1, x2 = i[0], i[1]
        a, b = r2plot[x1], r2plot[x2]
        if len(a)==len(b):
            res = wilcoxon(a,b)
        else:
            res = mannwhitneyu(a, b)

        if res[1] < 0.05:
            label = '*'

        if res[1] < 0.001:
            label = '**'

        # if res[1] < 0.0001:
        #     label = '***'

        if res[1] > 0.05:
            label = "ns"

        y0 = .8 #np.max(np.concatenate((r2plot[x1], r2plot[x2])))
        y, h, col = y0 + (x1 + x2)/12, .025, 'k'
        # x1 +=1
        # x2 +=1
        plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
        plt.text((x1+x2)*.5, y+h, label, ha='center', va='bottom', color=col)


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(0, len(labels)))
    ax.set_xticklabels(labels, rotation=-45, ha='left', rotation_mode='anchor')
    ax.set_xlim(-0.75, len(labels) - 0.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_xlabel('Model')

r2 = []
modellist = ['stimdrift', 'offset', 'affine']
for f in modellist:
    r2.append(np.concatenate([d['r2models'][f] for d in dstats]))

modellist = ['Stim. + Slow drift', 'Shared Offset', 'Shared Affine']
goodix = r2[-1] > 0 # , ~np.isnan(r2[3]))
goodix = np.logical_and(goodix, r2[1]>0)
# goodix = r2[0] > -np.inf

clr_mouse = np.asarray([206, 110, 41])/255
clr_marmoset = np.asarray([51, 121, 169])/255
import seaborn as sns

plt.figure(figsize=(5,5))
ax = plt.subplot(1,2,1)
ix = np.logical_and(goodix, subjid==0)
r2plot = [r[ix] for r in r2]

fancy_violinplot(r2plot, modellist, clr_marmoset)
# plt.axhline(0, color='k')
plt.ylim([-0,1.2])
plt.ylabel('cv $r^2$')
# plt.title('Marmoset')

# sns.despine(trim=True, offset=0)

ax1 = plt.subplot(1,2,2)
ix = np.logical_and(goodix, subjid==1)
r2plot = [r[ix] for r in r2]
fancy_violinplot(r2plot, modellist, clr_mouse)
# plt.axhline(0, color='k')
plt.ylim([0,1.2])
set_axis_style(ax, modellist)
set_axis_style(ax1, modellist)
# plt.ylabel('cv $r^2$')
# plt.title('Mouse')

# sns.despine(trim=True, offset=0)
plt.savefig("../Figures/model_compare_violin.pdf")

from scipy.stats import wilcoxon


plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
i = 0 # stim drift
j = -1 # affine
ix = np.logical_and(goodix, subjid==1)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.8, color=clr_mouse, label='Mouse')
res = wilcoxon(r2[i][ix], r2[j][ix])
print('Mouse has %d units to analyze' %(sum(ix)))
if res[1] < 0.05:
    print("Mouse: Affine [%.4f] significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
else:
    print("Mouse: Affine [%.4f] NOT significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

ix = np.logical_and(goodix, subjid==0)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.8, color=clr_marmoset, label='Marmoset')
print('Marmoset has %d units to analyze' %(sum(ix)))
res = wilcoxon(r2[i][ix], r2[j][ix])
if res[1] < 0.05:
    print("Marmoset: Affine [%.4f] significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
else:
    print("Marmoset: Affine [%.4f] NOT significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

plt.plot([0,1], [0,1], 'k--')
plt.legend()
plt.xlabel(modellist[i])
plt.ylabel(modellist[j])
sns.despine(trim=True, offset=0)


plt.subplot(1,2,2)
i = 1
j = 2
ix = np.logical_and(goodix, subjid==1)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.5, color=clr_mouse, label='Mouse')
res = wilcoxon(r2[i][ix], r2[j][ix])
if res[1] < 0.05:
    print("Mouse: Affine [%.4f] significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
else:
    print("Mouse: Affine [%.4f] NOT significantly better than stimdrift [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

ix = np.logical_and(goodix, subjid==0)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.5, color=clr_marmoset, label='Marmoset')
res = wilcoxon(r2[i][ix], r2[j][ix])
if res[1] < 0.05:
    print("Marmoset: Affine [%.4f] significantly better than Offset [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
else:
    print("Marmoset: Affine [%.4f] NOT significantly better than Offset [%.4f] stat=%d, p=%d" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

plt.plot([0,1], [0,1], 'k--')
plt.xlabel(modellist[i])
plt.ylabel(modellist[j])
sns.despine(trim=True, offset=0)
plt.savefig("../Figures/model_compare_scatter.pdf")


#%% get null distributions
mouse_sessions = np.where(['mouse' in d['sess'] for d in dstats])[0]
marm_sessions = np.where(['marmoset' in d['sess'] for d in dstats])[0]

def get_pseudosession_null(dstats, field1, sesslist, field2='running'):
    """
    Get null distribution for a field using pseudo-session method
    """
    null = []
    for i in sesslist:
        for j in np.setdiff1d(sesslist, i):
            x = dstats[i][field1]
            y = dstats[j][field2]
            NT = np.minimum(len(x), len(y))
            res = spearmanr(x[:NT], y[:NT], nan_policy='omit')
            null.append(res[0])

    return np.asarray(null)

def run_stats_check(dstats, field1='gain', field2='running', plot_individual=True, label_significant=False, alpha=.5, num_bins=30):
    import warnings
    warnings.filterwarnings('ignore')
    import seaborn as sns    
    print("RUNNING COMPARISON")
    print("Comparing %s to %s" % (field1, field2))

    NCs = np.asarray([len(d['r2models']['gain']) for d in dstats])
    NTs = np.asarray([len(d['running']) for d in dstats])
    goodix = np.logical_and(NCs>10, NTs>300)
    mouse_sessions = np.where(np.logical_and(['mouse' in d['sess'] for d in dstats], goodix))[0]
    marm_sessions = np.where(np.logical_and(['marmoset' in d['sess'] for d in dstats], goodix))[0]

    mouse_null_z = get_pseudosession_null(dstats, field1, mouse_sessions, field2=field2)
    marm_null_z = get_pseudosession_null(dstats, field1, marm_sessions, field2=field2)

    mouse_r = np.asarray([spearmanr(dstats[i][field1], dstats[i][field2], nan_policy='omit')[0] for i in mouse_sessions])
    marm_r = np.asarray([spearmanr(dstats[i][field1], dstats[i][field2], nan_policy='omit')[0] for i in marm_sessions])

    # compare real data to null
    pvalmouse = np.mean(mouse_r[:,None] > mouse_null_z, axis=1)
    pvalmouse[pvalmouse>.5] = 1 - pvalmouse[pvalmouse>.50]

    pvalmarm = np.mean(marm_r[:,None] > marm_null_z, axis=1)
    pvalmarm[pvalmarm>.5] = 1 - pvalmarm[pvalmarm>.50]

    # scale for two-sided?
    pvalmouse *= 2
    pvalmarm *= 2

    if plot_individual:
        plt.figure()
        plt.plot(marm_sessions, pvalmarm, 'o', label='marmoset')
        plt.plot(mouse_sessions, pvalmouse, 'o', label='mouse')
        plt.axhline(0.05, color='k')
        plt.ylabel('p-value')
        plt.xlabel('Session')
        plt.legend()

    plt.figure(figsize=(2.5,5))
    plt.subplot(2,1,1)
    bins = np.linspace(-1,1,num_bins)
    f = plt.hist(marm_r, bins=bins, color=clr_marmoset, alpha=alpha, label='marmoset')
    if label_significant:
        plt.hist(marm_r[pvalmarm<0.05], bins=bins, color=clr_marmoset, alpha=1, label='significant')
    m_marm = np.nanmedian(marm_r)
    res = wilcoxon(marm_r)
    plt.plot(m_marm, f[0].max()+1, 'v', color=clr_marmoset)
    if res[1] < 0.05:
        plt.plot(m_marm, f[0].max()+2, '*k')
        print('Marmoset: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.3f' %(m_marm, res[0], res[1]))
    else:
        print('Marmoset: %.3f not significantly different 0. Wilcoxon stat (%d), p=%.3f' %(m_marm, res[0], res[1]))
        
    plt.ylabel('Num. sessions')
    plt.title("%s vs. %s" % (field1, field2))
    plt.xlim(-1,1)

    # plt.title('Marmoset')
    plt.subplot(2,1,2)
    bins = np.linspace(-1,1,num_bins)
    f = plt.hist(mouse_r, bins=bins, color=clr_mouse, alpha=alpha, label='marmoset')
    if label_significant:
        plt.hist(mouse_r[pvalmouse<0.05], bins=bins, color=clr_mouse, alpha=1, label='significant')
    m_mouse = np.nanmedian(mouse_r)
    res = wilcoxon(mouse_r)
    plt.plot(m_mouse, f[0].max()+1, 'v', color=clr_mouse)
    if res[1] < 0.05:
        plt.plot(m_mouse, f[0].max()+2, '*k')
        print('Mouse: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.3f' %(m_mouse, res[0], res[1]))
    else:
        print('Mouse: %.3f not significantly different 0. Wilcoxon stat (%d), p=%.3f' %(m_mouse, res[0], res[1]))

    plt.xlabel('Spearman r')
    plt.xlim(-1,1)
    sns.despine(trim=True)
    plt.ylabel('Num. sessions')

    res = mannwhitneyu(marm_r[~np.isnan(marm_r)], mouse_r[~np.isnan(mouse_r)])
    if res[1] < 0.05:
        print('Marmoset vs. Mouse significantly different than eachother. Mann-Whitney stat (%d), p=%.7f' %(res[0], res[1]))
    return marm_r, mouse_r

#%% plot stats check
marm_rg, mouse_rg = run_stats_check(dstats, field1='zg', field2='running', plot_individual=False, alpha=.6, num_bins=25, label_significant=True)
plt.savefig("../Figures/corr_running_gain.pdf")

run_stats_check(dstats, field1='zg', field2='pupil', plot_individual=False, alpha=.6, num_bins=25, label_significant=True)

marm_rh, mouse_rh = run_stats_check(dstats, field1='zh', field2='running', plot_individual=False, alpha=.6, num_bins=25, label_significant=True)
plt.savefig("../Figures/corr_running_offset.pdf")

run_stats_check(dstats, field1='zh', field2='pupil', plot_individual=False, alpha=.6, num_bins=25, label_significant=True)

#%% plot gain vs. offset

plt.figure(figsize=(3,3))
plt.axhline(0, color='k')
plt.axvline(0, color='k')
plt.xlim(-1,1)
plt.ylim(-1,1)

plt.plot(marm_rg, marm_rh, 'o', color=clr_marmoset, markersize=5)
plt.plot(mouse_rg, mouse_rh, 'o', color=clr_mouse, markersize=5)
plt.xlabel('Gain (Spearman r)')
plt.ylabel('Offset (Spearman r)')
plt.title('Correlation w/ Running')
sns.despine(trim=True)
plt.savefig('../Figures/corre_running_2D.pdf')
#%% measure of Total gain fluctuations


bins = np.linspace(np.min(gainrange), np.max(gainrange), 50)

plt.figure(figsize=(3,3))
ax = plt.subplot()
cmap = plt.cm.get_cmap('tab10')
f1 = plt.hist(gainrange[subjid==1], bins=bins, alpha=1, label='mouse', color=clr_mouse, density=True)
f2 = plt.hist(gainrange[subjid==0], bins=bins, alpha=0.8, label='marmoset', color=clr_marmoset, density=True)
plt.legend()
plt.xlim(0,5)
m1 = np.nanmedian(gainrange[subjid==1])
m2 = np.nanmedian(gainrange[subjid==0])

res = mannwhitneyu(gainrange[subjid==1], gainrange[subjid==0])

my = np.max(plt.ylim())

plt.xlabel('std. (gain modulation)')
plt.ylabel('Density')
plt.plot(m1, my, 'v', color=clr_mouse)
plt.plot(m2, my, 'v', color=clr_marmoset)
sns.despine(trim=True)
plt.savefig('../Figures/gain_range.pdf')
print('mouse modulates by {}. marmoset by {}'.format(m1, m2))


#%% Correlation with different variables
gainrunrho = np.asarray([d['gainruncorr'][0] for d in dstats])
gainrunpval = np.asarray([d['gainruncorr'][1] for d in dstats])
subix = np.asarray(['mouse' in d['sess'] for d in dstats])

sessid = np.arange(len(gainrunrho))
sig = gainrunpval < 0.05

plt.plot(sessid[subix], gainrunrho[subix], 'o', alpha=0.5, color=clr_mouse)
plt.plot(sessid[~subix], gainrunrho[~subix], 'o', alpha=0.5, color=clr_marmoset)
ix = np.logical_and(sig, subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=clr_mouse, label='Mouse')
ix = np.logical_and(sig, ~subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=clr_marmoset, label='Marmoset')

plt.ylabel('Correlation with Running')
plt.axhline(0, color='k')
plt.xlabel('Session ID')
plt.legend()

#%% Correlation with pupil
gainrunrho = np.asarray([d['gainpupilcorr'][0] for d in dstats])
gainrunpval = np.asarray([d['gainpupilcorr'][1] for d in dstats])
subix = np.asarray(['mouse' in d['sess'] for d in dstats])

sessid = np.arange(len(gainrunrho))
sig = gainrunpval < 0.05

plt.plot(sessid[subix], gainrunrho[subix], 'o', alpha=0.5, color=clr_mouse)
plt.plot(sessid[~subix], gainrunrho[~subix], 'o', alpha=0.5, color=clr_marmoset)
ix = np.logical_and(sig, subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=clr_mouse, label='Mouse')
ix = np.logical_and(sig, ~subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=clr_marmoset, label='Marmoset')

plt.ylabel('Correlation with Pupil')
plt.axhline(0, color='k')
plt.xlabel('Session ID')
plt.legend()

#%% Correlation between pupil and running
gainrunrho = np.asarray([d['runningpupilcorr'][0] for d in dstats])
gainrunpval = np.asarray([d['runningpupilcorr'][1] for d in dstats])
subix = np.asarray(['mouse' in d['sess'] for d in dstats])

sessid = np.arange(len(gainrunrho))
sig = gainrunpval < 0.05

plt.plot(sessid[subix], gainrunrho[subix], 'o', alpha=0.5, color=cmap(0))
plt.plot(sessid[~subix], gainrunrho[~subix], 'o', alpha=0.5, color=cmap(2))
ix = np.logical_and(sig, subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=cmap(0), label='Mouse')
ix = np.logical_and(sig, ~subix)
plt.plot(sessid[ix], gainrunrho[ix], 'o', alpha=1.0, color=cmap(2), label='Marmoset')

plt.ylabel('Correlation btw. Running and Pupil')
plt.axhline(0, color='k')
plt.xlabel('Session ID')
plt.legend()


#%% plot CV PCA



for subject in ['mouse', 'marmoset']:

    
    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)
    
        N = len(das['cvpca'])
        figures.append(plt.figure())
        NC = len(das['cvpca'][0]['r2test'])
        r2train = [np.mean(d['r2train']) for d in das['cvpca']]
        r2test = [np.mean(d['r2test']) for d in das['cvpca']]

        for i in range(N):
            plt.plot(i*np.ones(NC), das['cvpca'][i]['r2test'], '.', color='gray')
        plt.plot(r2train, '-o', label='Train')
        plt.plot(r2test, '-o', label='Test')
        plt.xlabel("# Dimensions")
        plt.ylabel("Var. Explained")
        plt.ylim(-.4, 1)
        plt.axhline(0, linestyle='--', color='black')
        tstr = aname.replace('.pkl', '') + ": %d NC, %d NT" %(NC, das['data']['robs'].shape[0])
        plt.title(tstr)
        
    
import matplotlib.backends.backend_pdf
pdf = matplotlib.backends.backend_pdf.PdfPages("latent_dims.pdf")
for fig in figures: ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()


#%% plot single session
%matplotlib inline
figs = []
import torch.nn.functional as F

aname = 'marmoset_14.pkl'
fname = aname.replace('.pkl', '.mat')

sortgain = True

if 'mouse' in aname:
    clrmap = 'Reds'
else:
    clrmap = 'Blues'

print(aname)

refit = False

if refit:
    a = fit_session(fpath, apath, fname, aname, ntents=5, seed=1234)

with open(apath + aname, 'rb') as f:
    das = pickle.load(f)

plt.figure(figsize=(10,5))
plot_summary(das, aname, modelname='affine')

# plot nicely
plt.figure(figsize=(10,5))

model = das['affine']['model']
zgain = model.gain_mu.get_weights()
zweight = model.readout_gain.get_weights()
if np.median(zweight) < 0: # flip sign if both are negative
    zgain *= -1
    zweight *= -1
# if hasattr('')
zoffset = model.offset_mu.get_weights()
zoffweight = model.readout_offset.get_weights()
if np.median(zoffweight) < 0: # flip sign if both are negative
    zoffset *= -1
    zoffweight *= -1

dfs = das['data']['dfs'][:,das['affine']['model'].cids]
robs = das['data']['robs'][:,das['affine']['model'].cids]*dfs
# robs = robs[:,das['affine']['r2test']>0]

if sortgain:
    ind = np.argsort(zweight)
else:
    ind = np.argsort(zoffweight)

robs = robs[:,ind]
axspikes = plt.subplot(2,1,1)
plt.imshow(np.sqrt(robs.T), aspect='auto', interpolation='none', cmap=clrmap)
axspikes.get_xaxis().set_ticks([500])
axspikes.get_yaxis().set_ticks([0,50])
sns.despine(ax=axspikes,trim=True)
xd = (00, 400)

plt.xlim(xd)

axrun = plt.subplot(6,1,4)

plt.plot(das['data']['running'], 'k')
# plt.axhline(0, color=[.5, .5, .5])


axrun.spines['bottom'].set_visible(False)
axrun.get_xaxis().set_ticks([500])

sns.despine(ax=axrun,trim=True)
plt.xlim(xd)
# axgain.get_yaxis().set_ticks([])

axgain = plt.subplot(6,1,5)
axgain.plot(zgain, 'g')
axgain.axhline(0, color=[.5, .5, .5])
axgain.set_xlim(xd)
sns.despine(ax=axgain,trim=True)
plt.axis("off")

axoff = plt.subplot(6,1,6)
plt.plot(zoffset, 'm')
plt.axhline(0, color=[.5, .5, .5])
plt.xlim(xd)
# sns.despine(trim=True)
axoff.spines['left'].set_visible(False)
axoff.get_yaxis().set_ticks([0])
axoff.get_xaxis().set_ticks([0,50])
sns.despine(ax=axoff,trim=True)
# plt.axis("off")

# plt.xlabel('Trial')

plt.savefig('../Figures/example_affine_' + aname.replace('.pkl', '.pdf'))


#%% gain distributions


running = np.concatenate([d['running'] for d in dstats]).flatten()
gainsubjid = np.concatenate([ np.ones(len(d['running']))*('mouse' in d['sess']) for d in dstats]).flatten()
zglatent = np.concatenate([d['zg'] for d in dstats])
zhlatent = np.concatenate([d['zh'] for d in dstats])

# congert g to gain
zglatent = np.maximum(zglatent + 1, 0)

plt.figure(figsize=(10,5))
bins = np.linspace(0, 4, 50)

ax1 = plt.subplot(2,2,1)
ix = np.logical_and(running > 3, gainsubjid==1)
f = plt.hist(zglatent[ix], bins=bins, alpha=.5, density=True, label='running')
ix = np.logical_and(np.abs(running) < 3, gainsubjid==1)
f = plt.hist(zglatent[ix], bins=bins, alpha=.5, density=True, label='stationary')
plt.xlabel('$z_{gain}$')
ax1.patch.set_alpha(0.0)
ax1.legend()

ax2 = plt.subplot(2,2,2)
ix = np.logical_and(running > 3, gainsubjid==0)
f = plt.hist(zglatent[ix], bins=bins, alpha=.5, density=True)
ix = np.logical_and(np.abs(running) < 3, gainsubjid==0)
f = plt.hist(zglatent[ix], bins=bins, alpha=.5, density=True)
ax2.patch.set_alpha(0.0)
plt.xlabel('$z_{gain}$')

bins = np.linspace(-4, 4, 50)

ax3 = plt.subplot(2,2,3)
ix = np.logical_and(running > 3, gainsubjid==1)
f = plt.hist(zhlatent[ix], bins=bins, alpha=.5, density=True, label='running')
ix = np.logical_and(np.abs(running) < 3, gainsubjid==1)
f = plt.hist(zhlatent[ix], bins=bins, alpha=.5, density=True, label='stationary')
ax3.patch.set_alpha(0.0)
plt.xlabel('$z_{h}$')

ax4 = plt.subplot(2,2,4)
ix = np.logical_and(running > 3, gainsubjid==0)
f = plt.hist(zhlatent[ix], bins=bins, alpha=.5, density=True)
ix = np.logical_and(np.abs(running) < 3, gainsubjid==0)
f = plt.hist(zhlatent[ix], bins=bins, alpha=.5, density=True)
ax4.patch.set_alpha(0.0)
plt.xlabel('$z_{h}$')

sns.despine()


#%% get simple means for gain and offset
plt.figure(figsize=(2,3))
ax = plt.subplot()
run_thresh = 3
zgmur = np.asarray([np.mean(d['zg'][d['running']>run_thresh]) for d in dstats])
zgmus = np.asarray([np.mean(d['zg'][np.abs(d['running'])<run_thresh]) for d in dstats])
zhmur = np.asarray([np.mean(d['zh'][d['running']>run_thresh]) for d in dstats])
zhmus = np.asarray([np.mean(d['zh'][np.abs(d['running'])<run_thresh]) for d in dstats])
ixmouse = ['mouse' in d['sess'] for d in dstats]
x = ['zg (mouse)']*sum(ixmouse) + ['zg (marmoset)']*(len(ixmouse)-sum(ixmouse))
x = x + ['zh (mouse)']*sum(ixmouse) + ['zh (marmoset)']*(len(ixmouse)-sum(ixmouse))
y = list(zgmur-zgmus) + list(zhmur-zhmus)
sns.pointplot(x=x, y=y, join=False, palette=[clr_mouse, clr_marmoset, clr_mouse, clr_marmoset])
ax.set_xticklabels(['$z_g$ (mouse)', '$z_g$ (marmoset)', '$z_h$ (mouse)', '$z_h$ (marmoset)'], rotation=-45, ha='left', rotation_mode='anchor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.ylabel('Running - Stationary')
# plt.figure()
# plt.plot(zgmur-zgmus, 'o')
plt.axhline(0, color='gray')
plt.savefig('../Figures/latents_running_delta.pdf')



# plt.plot(zgmus, 'o')

#%%
ws1 = das['affine']['model'].stim.get_weights()
ws0 = das['stimdrift']['model'].stim.get_weights()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(ws0, aspect='auto', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(ws1, aspect='auto', interpolation='none')