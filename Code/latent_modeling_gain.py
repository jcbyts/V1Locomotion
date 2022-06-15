#%% Import
import os, sys

sys.path.insert(0, '/mnt/Data/Repos/')
sys.path.append("../")

import numpy as np

import matplotlib.pyplot as plt
plt.style.use('plot_style.txt')

import seaborn as sns

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
exclude_calcarine = ['marmoset_13.pkl', 'marmoset_15.pkl']

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

sess_success = np.zeros(nsessions)
sessinfo = []

for isess in range(0, nsessions):
    print(isess)
    fname = flist[isess]
    aname = fname.replace('.mat', '.pkl')

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
    fracrun = np.mean(das['data']['running']>3)
    nCids = len(das['affine']['model'].cids)
    sessinfo.append({'subj': subject, 'id': id, 'NT': NT, 'NC': NC, 'nCids': nCids, 'fracrun': fracrun, 'calcarine': aname in exclude_calcarine})


#%% Plot summary about number of neurons
subjs = ['marmoset', 'mouse']
output_file = '../output/latent_modling.txt'
fid = open(output_file, 'w+')
fid.write("****************************************************\r\n")
fid.write("****************************************************\r\n")
fid.write("LATENT MODELING\r\n")
fid.write("****************************************************\r\n\n\n\n")

fid.write("Overview of sessions:\n")

NT = []
NC = []
nCids = []
calcarine = []
fracrun = []

for subj in subjs:
    sessinds = np.where([s['subj'] == subj for s in sessinfo])[0]
    print("Found %d sessions for subject %s" % (len(sessinds), subj))
    
    NT.append(np.asarray([sessinfo[i]['NT'] for i in sessinds]))
    NC.append(np.asarray([sessinfo[i]['NC'] for i in sessinds]))
    calcarine.append(np.asarray([sessinfo[i]['calcarine'] for i in sessinds]))
    fracrun.append(np.asarray([sessinfo[i]['fracrun'] for i in sessinds]))
    nCids.append(np.asarray([sessinfo[i]['nCids'] for i in sessinds]))

    for isess in range(len(sessinds)):
        sess = sessinfo[sessinds[isess]]

        msg = "%d) %s %d, %d trials, %d units (%d good)" % (isess, sess['subj'], sess['id'], sess['NT'], sess['NC'], sess['nCids'])
        print(msg)
        fid.write(msg + '\r')


plt.figure(figsize=(10,4))

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


fid.write("\n\n")
for isubj, subj in enumerate(subjs):
    iix = NC[isubj] > neuron_thresh

    iix = np.logical_and(iix, NT[isubj]>trial_thresh)
    runningix =np.logical_and(fracrun[isubj] >= running_frac_thresh[0], fracrun[isubj] <= running_frac_thresh[1])
    iix = np.logical_and(iix, runningix)
    iix = np.logical_and(iix, ~calcarine[isubj])
    
    nsess = len(iix)

    fid.write("%s: %d/%d sessions included. %d/%d Units (%f units per session)\n\n" %(subj, sum(iix), nsess, sum(NC[isubj][iix]), sum(NC[isubj]), np.mean(NC[isubj][iix])))


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

def useaeifbetter(das):
    from copy import deepcopy
    modlist = ['offset', 'affine', 'gain']
    for mod in modlist:
        r2 = das[mod]['r2test']
        r2ae = das[mod + 'ae']['r2test']
        if r2ae.mean() > r2.mean():
            das[mod] = deepcopy(das[mod + 'ae'])
    return das


def plot_summary(das, aname, modelname='affine'):
    gridspec.GridSpec(2,3)


    model = das[modelname]['model']
    sz = das['data']['running'].shape
    if hasattr(model, 'gain_mu'):
        zgain = model.gain_mu.get_weights()
        zweight = model.readout_gain.get_weights()
        if np.mean(zweight) < 0: # flip sign if both are negative
            zgain *= -1
            zweight *= -1
    else:
        zgain = np.zeros(sz)
        zweight = np.zeros(len(model.cids))

    dfs = das['data']['dfs'][:,das[modelname]['model'].cids]
    robs = das['data']['robs'][:,das[modelname]['model'].cids]*dfs
    # robs = robs
    robs = (robs - np.min(robs, axis=0)) / (np.max(robs, axis=0) - np.min(robs, axis=0))

    ind = np.argsort(zweight)
    ax0 = plt.subplot2grid((6,3), (0,0), colspan=2, rowspan=3)

    plt.imshow(robs[:,ind].T, aspect='auto', interpolation='none', cmap='Blues')
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

    plt.xlabel("Trial")
    

    from scipy.stats import spearmanr
    rhog = spearmanr(das['data']['running'], zgain)

    titlestr = 'Corr w/ running: gain '
    titlestr += "%0.3f" %rhog[0]

    if rhog[1] < 0.05:
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

#%% plot session summaries

for subject in ['mouse', 'marmoset']:

    filelist = os.listdir(apath)
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]
        if aname in exclude_calcarine:
            continue

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)

        skip, msg = skip_session(das, neuron_thresh=neuron_thresh, trial_thresh=trial_thresh, run_frac=running_frac_thresh)
        das = useaeifbetter(das)

        if skip:
            print(msg)
            continue

        figures.append(plt.figure(figsize=(10,5)))
        plot_summary(das, aname, modelname='gain')
        

pdf = matplotlib.backends.backend_pdf.PdfPages("robs_and_gain.pdf")
for fig in figures: ## will open an empty extra figure :(
    pdf.savefig( fig )
pdf.close()

#%% plot model comparison summary
%matplotlib inline
clr_mouse = np.asarray([206, 110, 41])/255
clr_marmoset = np.asarray([51, 121, 169])/255

fig = plt.figure(figsize=(2,2))
ax = plt.subplot()

fig2 = plt.figure()
ax2 = plt.subplot()

r2s_stim = []
r2s_gain = []

for subject in ['mouse', 'marmoset']:
    
    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]

        if aname in exclude_calcarine:
            continue

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

        das = useaeifbetter(das)

        r2stim = das['stimdrift']['r2test'].mean().cpu()
        r2gain = das['gain']['r2test'].mean().cpu()

        r2s_gain.append(r2gain)
        r2s_stim.append(r2stim)

        ax.plot(r2stim, r2gain, '.', color=clr, markersize=10)
        
        nCids = len(das['gain']['model'].cids)

        NT = das['data']['robs'].shape[0]
        if fid is not None:
            fid.write("%s: NT=%d NC=%d r2stim = %.4f r2gain = %.4f\n" %(aname, NT, nCids, r2stim, r2gain))
        mdiff = r2gain - r2stim
        ax2.plot(NT, mdiff, '.', color=clr, markersize=10)

ax.plot((0,.8),(0,.8), 'k')
ax.set_xlabel('Stimulus + Slow Drift ($r^2$)')
ax.set_ylabel("Shared Gain ($r^2$)")
ax.set_xlim( ( 0, .8))
ax.set_ylim( ( 0, .8))
sns.despine(ax=ax, trim=True)
fig.savefig("../Figures/model_compare_session.pdf", bbox_inches="tight")

ax2.axhline(0, color='k')
# ax2.set_xlabel("# neurons")
ax2.set_xlabel("# trials")
ax2.set_ylabel('Model improvement with gain')
sns.despine(trim=True)


#%% more functions
from scipy.stats import spearmanr
import pickle

def dprime(x, ix):

    mu1 = x[ix.flatten(),:].mean(dim=0)
    mu2 = x[~ix.flatten(),:].mean(dim=0)
    sd = x.std(dim=0)
    dp = (mu1 - mu2) / sd
    return dp.mode()[0].item()

def sessstats(das, fname=None, modelname='gain',aefallback=True):
    
    import torch.nn.functional as F
    from copy import deepcopy

    if aefallback:
        das = useaeifbetter(das)
    
    running = das['data']['running']
    pupil = das['data']['pupil']
    mod = das[modelname]['model']
    mod.to('cpu')

    robs = torch.tensor(das['data']['robs'])

    # latent gain gets datafiltered input
    dfs = das['data']['dfs']
    robs = robs * dfs

    zg = mod.gain_mu.weight


    # get population level gains
    zgpop = F.relu(1 + mod.readout_gain(zg).detach().cpu())
    zglatent = zgpop.mean(dim=1).detach().cpu() # the average gain across the population (in units of gain)
    
    zg = zg.detach().cpu() # the latent (arbitrary units)

    res = spearmanr(running, pupil, nan_policy='omit')

    dstat = {}
    dstat['sess'] = '%s %i' %(subject, isess)
    dstat['fname'] = fname
    dstat['gainrange'] = zgpop.std(dim=0).numpy()
    dstat['runningpupilcorr'] = res
    dstat['gainruncorr'] = spearmanr(zglatent, running, nan_policy='omit')
    dstat['gainpupilcorr'] = spearmanr(zglatent, pupil, nan_policy='omit')
    dstat['running'] = running
    dstat['pupil'] = pupil
    dstat['zglatent'] = zglatent.numpy()
    dstat['zg'] = zg.numpy()
    dstat['zgflip'] = zg.numpy() * np.sign(np.mean(mod.readout_gain.get_weights()))

    modellist = ['drift', 'stimdrift', 'offset', 'gain', 'affine', 'affineae']
    dstat['r2models'] = {}
    for f in modellist:
        dstat['r2models'][f] = das[f]['r2test'].numpy()

    return dstat


#% Model performance plotting
def fancy_violinplot(r2plot, modellist, clr, fid=None, bw='scott', inner='quartile'):
    import seaborn as sns
    from scipy.stats import mannwhitneyu
    
    # sns.violinplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]), y=np.concatenate(r2plot),
        # color=clr, bw=bw, inner=inner, cut=0)
    
    if fid is not None:
        fid.write("\nfancy_violinplot called\n")

    sns.stripplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]),
        y=np.concatenate(r2plot),
        color=clr, size=1.5, dodge=True, jitter=.35, alpha=.5, zorder=1)

    sns.pointplot(x=np.concatenate([len(r2plot[i])*[modellist[i]] for i in range(len(modellist))]), y=np.concatenate(r2plot),
        color='black', estimator=np.median, markers='o', scale=.5, dodge=True)
    

    n = len(r2plot)
    import itertools
    for i in itertools.combinations(range(n),2):
        x1, x2 = i[0], i[1]
        a, b = r2plot[x1], r2plot[x2]
        if len(a)==len(b):
            res = wilcoxon(a,b)
            if fid is not None:
                fid.write("Comparing %s and %s\n" %(modellist[x1], modellist[x2]))
                fid.write("Wilcoxon signed-rank test: stat=%f, p=%.2e\n" %(res[0], res[1]))
        else:
            res = mannwhitneyu(a, b)
            if fid is not None:
                fid.write("Comparing %s and %s\n" %(modellist[x1], modellist[x2]))
                fid.write("Mann-Whitney U test: stat=%f, p=%.2e\n" %(res[0], res[1]))
                
        if res[1] < 0.05:
            label = '*'

        if res[1] < 0.001:
            label = '**'

        if res[1] > 0.05:
            label = "n.s."

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

#%% get session summaries
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

        if aname in exclude_calcarine:
            continue

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)

        dstats.append(sessstats(das, aname, modelname='gain'))

gainrange = np.concatenate([d['gainrange'] for d in dstats])
subjid = np.concatenate([ np.ones(len(d['gainrange']))*('mouse' in d['sess']) for d in dstats])
NCs = np.concatenate([len(d['r2models']['gain'])*[len(d['r2models']['gain'])] for d in dstats])
NTs = np.concatenate([len(d['r2models']['gain'])*[len(d['running'])] for d in dstats])

r2 = []
modellist = ['stimdrift', 'gain']
for f in modellist:
    r2.append(np.concatenate([d['r2models'][f] for d in dstats]))

modellist = ['Stim. + Slow drift', 'Shared Gain']
goodix = r2[0] > 0 # , ~np.isnan(r2[3]))
goodix = np.logical_and(goodix, r2[1]>0)

clr_mouse = np.asarray([206, 110, 41])/255
clr_marmoset = np.asarray([51, 121, 169])/255
import seaborn as sns

plt.figure(figsize=(2.5,2.1))
ax = plt.subplot(1,2,1)
ix = np.logical_and(goodix, subjid==0)
r2plot = [r[ix] for r in r2]

if fid is not None:
    fid.write("Calling violin plot for marmoset")
fancy_violinplot(r2plot, modellist, clr_marmoset, fid=fid)
# plt.axhline(0, color='k')
plt.ylim([-0,1.0])
plt.ylabel('cv $r^2$')

ax1 = plt.subplot(1,2,2)
ix = np.logical_and(goodix, subjid==1)
r2plot = [r[ix] for r in r2]
if fid is not None:
    fid.write("Calling violin plot for mouse")
fancy_violinplot(r2plot, modellist, clr_mouse, fid=fid)

plt.ylim([0,1.0])
set_axis_style(ax, modellist)
set_axis_style(ax1, modellist)
ax1.set_yticklabels([])
plt.subplots_adjust()

plt.savefig("../Figures/model_compare_violin.pdf", bbox_inches="tight")

#%%
from scipy.stats import wilcoxon

plt.figure(figsize=(2.5,2.5))
plt.subplot()
i = 0 # stim drift
j = 1 # gain
ix = np.logical_and(goodix, subjid==1)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.8, color=clr_mouse, label='Mouse')
res = wilcoxon(r2[i][ix], r2[j][ix])
if fid is not None:
    fid.write("Analyzing individual units again\n")
    fid.write('\nMouse has %d units to analyze\n' %(sum(ix)))
    if res[1] < 0.05:
        fid.write("Mouse: Gain [%.4f] significantly better than stimdrift [%.4f] stat=%d, p=%.2e\n" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
    else:
        fid.write("Mouse: Gain [%.4f] NOT significantly better than stimdrift [%.4f] stat=%d, p=%.2e\n" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

ix = np.logical_and(goodix, subjid==0)
plt.plot(r2[i][ix], r2[j][ix], '.', alpha=0.8, color=clr_marmoset, label='Marmoset')
res = wilcoxon(r2[i][ix], r2[j][ix])
if fid is not None:
    fid.write('\nMarmoset has %d units to analyze\n' %(sum(ix)))

    if res[1] < 0.05:
        fid.write("Marmoset: Gain [%.4f] significantly better than stimdrift [%.4f] stat=%d, p=%.2e\n" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))
    else:
        fid.write("Marmoset: Gain [%.4f] NOT significantly better than stimdrift [%.4f] stat=%d, p=%.2e\n" %(np.median(r2[j][ix]), np.median(r2[i][ix]), res[0], res[1]))

plt.plot([0,1], [0,1], 'k--')
plt.legend()
plt.xlabel(modellist[i])
plt.ylabel(modellist[j])
sns.despine(trim=True, offset=0)
plt.xlim(-.1, 1)
plt.ylim(-.1, 1)

plt.savefig("../Figures/model_compare_scatter.pdf", bbox_inches="tight")


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

def run_stats_check(dstats, field1='zgain', field2='running', plot_individual=True, label_significant=False, alpha=.5, num_bins=30, fid=None):
    import warnings
    warnings.filterwarnings('ignore')
    import seaborn as sns    
    
    if fid is None:
        print("RUNNING COMPARISON")
        print("Comparing %s to %s" % (field1, field2))
    else:
        fid.write("RUNNING COMPARISON with run_stats_check\n")
        fid.write("Comparing %s to %s\n" % (field1, field2))

    NCs = np.asarray([len(d['r2models']['affine']) for d in dstats])
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

    if fid is not None:
        fid.write("Comparing mouse to pseudo-session null\n")
        fid.write("%d/%d significant sessions\n" %(sum(pvalmouse < 0.05), len(pvalmouse)))
        
        fid.write("Comparing Marmoset to pseudo-session null\n")
        fid.write("%d/%d significant sessions\n" %(sum(pvalmarm < 0.05), len(pvalmarm)))

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
        plt.text(m_marm, f[0].max()+2, '*', ha='center', size=14)
        # plt.plot(m_marm, f[0].max()+2, '*k')
        if fid is not None:
            fid.write('Marmoset: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.2e\n' %(m_marm, res[0], res[1]))
        else:
            print('Marmoset: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.3f' %(m_marm, res[0], res[1]))
    else:
        if fid is not None:
            fid.write('Marmoset: %.3f not significantly different 0. Wilcoxon stat (%d), p=%.2e\n' %(m_marm, res[0], res[1]))
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
        plt.text(m_mouse, f[0].max()+2, '*', ha='center', size=14)
        # plt.plot(m_mouse, f[0].max()+2, '*k')
        if fid is not None:
            fid.write('Mouse: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.2e\n' %(m_mouse, res[0], res[1]))
        else:
            print('Mouse: %.3f significantly different than 0. Wilcoxon stat (%d), p=%.3f' %(m_mouse, res[0], res[1]))
    else:
        if fid is not None:
            fid.write('Mouse: %.3f not significantly different 0. Wilcoxon stat (%d), p=%.2e\n' %(m_mouse, res[0], res[1]))
        else:
            print('Mouse: %.3f not significantly different 0. Wilcoxon stat (%d), p=%.3f' %(m_mouse, res[0], res[1]))

    plt.xlabel('Spearman r')
    plt.xlim(-1,1)
    sns.despine(trim=True)
    plt.ylabel('Num. sessions')

    res = mannwhitneyu(marm_r[~np.isnan(marm_r)], mouse_r[~np.isnan(mouse_r)])
    if res[1] < 0.05:
        if fid is not None:
            fid.write('Marmoset vs. Mouse significantly different than eachother. Mann-Whitney stat (%d), p=%.2e\n' %(res[0], res[1]))
        else:
            print('Marmoset vs. Mouse significantly different than eachother. Mann-Whitney stat (%d), p=%.7f' %(res[0], res[1]))
    else:
        if fid is not None:
            fid.write('Marmoset vs. Mouse NOT significantly different than eachother. Mann-Whitney stat (%d), p=%.2e\n' %(res[0], res[1]))
        else:
            print('Marmoset vs. Mouse NOT significantly different than eachother. Mann-Whitney stat (%d), p=%.7f' %(res[0], res[1]))

    return marm_r, mouse_r

#%% plot stats check
marm_rg, mouse_rg = run_stats_check(dstats, field1='zg', field2='running', plot_individual=False, alpha=.6, num_bins=25, label_significant=True, fid=fid)
plt.savefig("../Figures/corr_running_gain.pdf", bbox_inches="tight")

run_stats_check(dstats, field1='zg', field2='pupil', plot_individual=False, alpha=.6, num_bins=25, label_significant=True, fid=fid)

#%% measure of Total gain fluctuations
from scipy.stats import bootstrap
bins = np.linspace(np.min(gainrange), np.max(gainrange), 60)

plt.figure(figsize=(3,3))
ax = plt.subplot()
cmap = plt.cm.get_cmap('tab10')
f1 = plt.hist(gainrange[subjid==1], bins=bins, alpha=1, label='mouse', color=clr_mouse, density=True)
f2 = plt.hist(gainrange[subjid==0], bins=bins, alpha=0.8, label='marmoset', color=clr_marmoset, density=True)
# plt.legend()
plt.xlim(0,8)
m1 = np.nanmedian(gainrange[subjid==1])
m2 = np.nanmedian(gainrange[subjid==0])



res = mannwhitneyu(gainrange[subjid==1], gainrange[subjid==0])

fid.write("\n\nComparing mouse and marmoset std. gain modulation\n")
fid.write("Mann Whitney U Test\n")
fid.write("p=%.3e, stat=%f\n" %(res[0], res[1]))

my = np.max(plt.ylim())

plt.xlabel('std. (gain modulation)')
plt.ylabel('Density')
plt.plot(m1, my, 'v', color=clr_mouse, markersize=10)
plt.plot(m2, my, 'v', color=clr_marmoset, markersize=10)
sns.despine(trim=True)
plt.savefig('../Figures/gain_range.pdf', bbox_inches="tight")

ci1 = bootstrap((gainrange[subjid==1],), np.nanmedian, n_resamples=1000)
ci2 = bootstrap((gainrange[subjid==0],), np.nanmedian, n_resamples=1000)

fid.write('mouse modulates by {} [{}, {}]. marmoset by {} [{}, {}]'.format(m1, ci1.confidence_interval[0], ci1.confidence_interval[1], m2, ci2.confidence_interval[0], ci2.confidence_interval[1]))


#%% get simple means for gain and offset
plt.figure(figsize=(1,3))
ax = plt.subplot()
run_thresh = 3
zgmur = np.asarray([np.mean(d['zg'][d['running']>run_thresh]) for d in dstats])
zgmus = np.asarray([np.mean(d['zg'][np.abs(d['running'])<run_thresh]) for d in dstats])

ixmouse = ['mouse' in d['sess'] for d in dstats]
x = ['g (mouse)']*sum(ixmouse) + ['g (marmoset)']*(len(ixmouse)-sum(ixmouse))

y = list(zgmur-zgmus)
sns.pointplot(x=x, y=y, join=False, palette=[clr_mouse, clr_marmoset, clr_mouse, clr_marmoset], estimator=np.mean, scale=.75)
ax.set_xticklabels(['g (mouse)', 'g (marmoset)'], rotation=-45, ha='left', rotation_mode='anchor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.ylabel('Latent Variable Difference\n Running - Stationary (a.u.)')

if fid is not None:
    fid.write("\nAnalyzing average difference in the gain latent variable\n")

from scipy.stats import ttest_1samp
groups = ['g (mouse)', 'g (marmoset)']
for i,group in enumerate(groups):
    
    ix = group == np.asarray(x)
    d = np.asarray(y)[ix]
    res = ttest_1samp(d, 0)
    
    ci = bootstrap((d,), np.mean, n_resamples=1000)
    mn = np.mean(d)
    if fid is not None:
        fid.write("%s mean difference: %.3f [%.3f, %.3f]\n" %(group, mn, ci.confidence_interval[0], ci.confidence_interval[1]))
        fid.write("%s ttest stat %.3f, p=%e\n" %(group, res[0], res[1]))

    if res[1] < 0.05:
        ax.text(i, np.mean(d) + .22 + 2*np.std(d)/np.sqrt(len(d)), '*', ha='center', size=14)
        # plt.plot(i, np.mean(d) + .22 + 2*np.std(d)/np.sqrt(len(d)), '*k', markersize=10)

plt.xlim(-1, 2)
# plt.figure()
# plt.plot(zgmur-zgmus, 'o')
plt.axhline(0, color='gray')
plt.savefig('../Figures/latents_running_delta.pdf', bbox_inches="tight")

#%% same but in gain units
plt.figure(figsize=(1,3))
ax = plt.subplot()
run_thresh = 3
zgmur = np.asarray([np.mean(d['zglatent'][:,None][d['running']>run_thresh]) for d in dstats])
zgmus = np.asarray([np.mean(d['zglatent'][:,None][np.abs(d['running'])<run_thresh]) for d in dstats])

ixmouse = ['mouse' in d['sess'] for d in dstats]
x = ['g (mouse)']*sum(ixmouse) + ['g (marmoset)']*(len(ixmouse)-sum(ixmouse))

y = list(zgmur-zgmus)
sns.pointplot(x=x, y=y, join=False, palette=[clr_mouse, clr_marmoset, clr_mouse, clr_marmoset], estimator=np.mean, scale=.75)
ax.set_xticklabels(['g (mouse)', 'g (marmoset)'], rotation=-45, ha='left', rotation_mode='anchor')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
plt.ylabel('Gain Difference\n Running - Stationary')

if fid is not None:
    fid.write("\nAnalyzing average gain difference after converting to gain\n")

from scipy.stats import ttest_1samp, ttest_ind
groups = ['g (mouse)', 'g (marmoset)']

dgroup = []
for i,group in enumerate(groups):
    
    ix = group == np.asarray(x)
    d = np.asarray(y)[ix]
    dgroup.append(d)
    res = ttest_1samp(d, 0)
    ci = bootstrap((d,), np.mean, n_resamples=1000)
    mn = np.mean(d)
    
    if fid is not None:
        fid.write("%s mean difference: %.3f [%.3f, %.3f]\n" %(group, mn, ci.confidence_interval[0], ci.confidence_interval[1]))
        fid.write("%s ttest stat %.3f, p=%e\n" %(group, res[0], res[1]))
    if res[1] < 0.05:
        ax.text(i, np.mean(d) + .1 + 2.5*np.std(d)/np.sqrt(len(d)), '*', ha='center', size=14)
        # plt.plot(i, np.mean(d) + .1 + 2.5*np.std(d)/np.sqrt(len(d)), '*k', markersize=10)


res = ttest_ind(dgroup[0], dgroup[1])
if fid is not None:
        fid.write("Comparing mouse and marmoset:\n")
        fid.write("2-sample ttest stat %.3f, p=%e\n" %(res[0], res[1]))


plt.xlim(-1, 2)
# plt.figure()
# plt.plot(zgmur-zgmus, 'o')
plt.axhline(0, color='gray')
plt.savefig('../Figures/latents_running_delta_avgain.pdf', bbox_inches="tight")

if fid is not None:
    fid.close()
    fid = None

#%% plot single session
%matplotlib inline
figs = []
import torch.nn.functional as F

aname = 'marmoset_8.pkl'
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
plot_summary(das, aname, modelname='gainae')

plt.figure(figsize=(10,5))
plot_summary(das, aname, modelname='gain')

das = useaeifbetter(das)

#%% plot nicely

fig = plt.figure(figsize=(4, 2))
plt.tight_layout(True)

model = das['gain']['model']

zweight = model.readout_gain.get_weights()
zgain = F.relu(1 + model.readout_gain(model.gain_mu.weight)).detach().cpu().numpy()
zgain = np.mean(zgain, axis=1)

zgain = model.gain_mu.get_weights()
zgain = np.maximum(zgain*np.mean(zweight) + 1, 0)
# if np.median(zweight) < 0: # flip sign if both are negative
#     zgain *= -1
#     zweight *= -1
# if hasattr('')

dfs = das['data']['dfs'][:,das['affine']['model'].cids]
robs = das['data']['robs'][:,das['affine']['model'].cids]*dfs

if sortgain:
    ind = np.argsort(zweight)
else:
    ind = np.arange(len(zweight))

robs = robs[:,ind]
robs = (robs - np.min(robs, axis=0)) / (np.max(robs, axis=0)- np.min(robs, axis=0))

# axspikes = plt.subplot(2,1,1)
axspikes = plt.axes((0.125, 0.55, 0.775, 0.55))
plt.imshow(robs.T, aspect='auto', interpolation='none', cmap=clrmap)
nc = robs.shape[1]
nt = np.minimum(robs.shape[0], 200)

plt.plot((0,50), (nc, nc), 'k', linewidth=2)
style = dict(size=6, color='black', ha='left')
axspikes.text(0, nc*1.1, "50 Trials", rotation=0, **style)
axspikes.text(nt*1.02, nc, "10 Neurons", rotation=90, **style)
plt.plot((nt,nt), (nc-10, nc), 'k', linewidth=4)

plt.axis("off")

xd = (00, nt)

plt.xlim(xd)

"""GAIN """
axgain = plt.subplot(4,1,3)
bbox = axgain.get_position()
bnds = bbox.bounds
nbnds = (bnds[0], bnds[1]+.05, bnds[2], bnds[3]*1.15)
axgain.set_position(nbnds)

axgain.axhline(1, color=[.5, .5, .5])
axgain.plot(zgain, color='#75AB53')

axgain.set_xlim(xd)

plt.axis("tight")
plt.axis("off")

axgain.set_xlim(xd)


"""Running"""
nbnds = (nbnds[0], nbnds[1]-.15, nbnds[2], nbnds[3]*.9)
axrun = plt.axes(nbnds)
# axrun = plt.subplot(4,1,4)
# bbox = axrun.get_position()
# bnds = bbox.bounds
# nbnds = (bnds[0], bnds[1]+0.03, bnds[2], bnds[3]*1)
# axrun.set_position(nbnds)

plt.plot(das['data']['running'], 'k')
spd = round(np.max(das['data']['running'])/2/5)*5

plt.plot((nt, nt), (0, spd), 'k', linewidth=4)
plt.axis("off")
axrun.text(nt*1.02, 0, "%d cm/s" %spd, rotation=90, **style)
axrun.set_xlim(xd)
plt.axis("tight")
axrun.set_xlim(xd)

# plt.subplots_adjust(top = .5, bottom = 0, right = 1, left = 0, 
            # hspace = 0, wspace = 0)
# plt.margins(0,0)
# fig.tight_layout()
# matplotlib.rcParams["figure.autolayout"] = False
#

# fig.set_tight_layout(True)

plt.savefig('../Figures/example_gain_' + aname.replace('.pkl', '.pdf'), bbox_inches="tight")


#%% gain distributions


running = np.concatenate([d['running'] for d in dstats]).flatten()
gainsubjid = np.concatenate([ np.ones(len(d['running']))*('mouse' in d['sess']) for d in dstats]).flatten()
zglatent = np.concatenate([d['zg'] for d in dstats])
zhlatent = np.concatenate([d['zh'] for d in dstats])

# congert g to gain
zglatent = np.maximum(zglatent + 1, 0)

plt.figure(figsize=(10,5))
bins = np.linspace(0, 10, 50)

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





#%% plot CV PCA

for subject in ['mouse', 'marmoset']:

    
    filelist = os.listdir(apath)
    # filelist = thelist
    filelist = [f for f in filelist if subject in f]

    for isess in range(len(filelist)):
        aname = filelist[isess]

        with open(apath + aname, 'rb') as f:
            das = pickle.load(f)

        das = useaeifbetter(das)
    
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


#%%
ws1 = das['affine']['model'].stim.get_weights()
ws0 = das['stimdrift']['model'].stim.get_weights()

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(ws0, aspect='auto', interpolation='none')
plt.subplot(1,2,2)
plt.imshow(ws1, aspect='auto', interpolation='none')


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
