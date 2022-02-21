import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.interpolate.interpolate import interp1d
from scipy.signal import savgol_filter

from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

class struct(dict):
    def __init__(self, *args, **kwargs):
        super(struct, self).__init__(*args, **kwargs)
        self.__dict__ = self

class GratingDataset(struct):
    def __init__(self,
        dataset_name,
        subject):

        self.dataset_name = dataset_name
        self.subject = subject

def get_huklab_sessions(data_directory = '/home/jake/Data/Datasets/HuklabTreadmill/processed/'):
    flist = [f for f in os.listdir(data_directory) if '_grat.mat' in f]
    return flist, data_directory

def get_huklab_session(data_directory, flist, session_id):
    import h5py
    fname = os.path.join(data_directory, flist[session_id])
    session = h5py.File(fname, 'r')
    return session

def process_huklab_session(session):
    # session is an h5py object
    import pandas as pd
    
    # get running speed
    run_time = session['treadTime'][0][:]
    run_spd = session['treadSpeed'][0][:]
    nans = np.isnan(run_spd)
    fint = interp1d(run_time[~nans], run_spd[~nans], kind='linear')
    new_time = np.arange(run_time[0], run_time[-1], 0.001)
    run_spd = fint(new_time)
    run_time = new_time
    D = struct()
    fpath, fname = os.path.split(session.filename)
    
    D.files = struct({'path': fpath, 'name': fname, 'session': os.path.splitext(fname)[0]})
    D.run_data = struct({'run_time': run_time,
        'run_spd': run_spd})

    D.eye_data = struct({'eye_time': session['eyeTime'][0][:],
        'eye_x': session['eyePos'][0][:],
        'eye_y': session['eyePos'][1][:],
        'pupil': session['eyePos'][2][:],
        'fs': 1//np.median(np.diff(session['eyeTime'][0][:])),
        'fs_orig': 1//np.median(np.diff(session['eyeTime'][0][:]))})

    D.saccades = detect_saccades(D.eye_data, vel_thresh=15, accel_thresh=5000, r2_thresh=-1e3, win=30, filter_length=51, debug=False)

    D.run_epochs = get_run_epochs(D.run_data.run_time, D.run_data.run_spd, debug=False)

    # organize grating data as a dataframe
    gratings = {'contrast': session['GratingContrast'][0][:],
        'orientation': session['GratingDirections'][0][:],
        'spatial_frequency': session['GratingFrequency'][0][:],
        'start_time': session['GratingOnsets'][0][:],
        'temporal_frequency': session['GratingFrequency'][0][:] * session['GratingSpeeds'][0][:],
        'duration': session['GratingOffsets'][0][:] - session['GratingOnsets'][0][:],
        'stop_time': session['GratingOffsets'][0][:]}

    stimulus_name = ['drifting_gratings' for i in range(len(gratings['contrast']))]
    phase = session['framePhase'][0][np.digitize(gratings['start_time'], session['frameTimes'][0])]

    conds = np.concatenate( (gratings['contrast'], gratings['orientation'], gratings['spatial_frequency'], gratings['temporal_frequency']), axis=0).reshape( (-1, len(gratings['contrast']))).T
    stimulus_condition_id = np.zeros(conds.shape[0])
    unique_conds = np.unique(conds, axis=0)
    for i in range(len(unique_conds)):
        ix = np.all((unique_conds[i,:] - conds)**2 < .001, axis=1)
        stimulus_condition_id[ix] = i

    stimulus_block = [1 for i in range(len(gratings['contrast']))]

    gratings['stimulus_name'] = stimulus_name
    gratings['phase'] = phase
    gratings['stimulus_condition_id'] = stimulus_condition_id
    gratings['stimulus_block'] = stimulus_block

    df = pd.DataFrame(gratings)
    df.index.name = 'stimulus_presentation_id'

    D['gratings'] = df

    # get spike data
    NC = len(session['units']['srf'])
    cids = [session[session['units']['id'][cc][0]][0][0].astype(int) for cc in range(NC)]
    az_rf = [session[session['units']['mu'][cc,0]][0].item() for cc in range(NC)]
    el_rf = [session[session['units']['mu'][cc,0]][1].item() for cc in range(NC)]

    spikes = dict()
    for cid in cids:
        spikes[cid] = session['spikeTimes'][0,session['spikeIds'][0,:]==cid]

    units = pd.DataFrame({'unit_id': cids,
    'ecephys_structure_acronym': ['VISp' for i in range(NC)],
    'rf_mat': [session[session['units']['srf'][cc][0]][:,:] for cc in range(NC)],
    'xax': [session[session['units']['xax'][cc][0]][:,:] for cc in range(NC)],
    'yax': [session[session['units']['yax'][cc][0]][:,:] for cc in range(NC)],
    'azimuth_rf': np.asarray(az_rf),
    'elevation_rf': np.asarray(el_rf)})

    units = units.set_index('unit_id')

    D['spikes'] = spikes
    D['units'] = units

    return D

# get "sessions"
def get_allen_sessions(data_directory = '/mnt/Data/Datasets/allen/ecephys_cache_dir/'):
    manifest_path = os.path.join(data_directory, "manifest.json")
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)
    sessions = cache.get_session_table()
    return sessions, cache

def get_allen_session(cache, sessions, session_id):
    return cache.get_session_data(sessions.index.values[session_id])
    
def process_allen_dataset(session,
    bin_size=0.01):

    D = struct()

    gazedata = session.get_screen_gaze_data()
    
    D['files'] = struct({'path': None, 'name': None, 'session': 'mouse_' + str(session.metadata['ecephys_session_id'])})
    D['eye_data'] = preprocess_gaze_data(gazedata)

    run_time = session.running_speed["start_time"] + \
    (session.running_speed["end_time"] - session.running_speed["start_time"]) / 2
    run_spd = session.running_speed['velocity'].values

    D['run_data'] = struct({'run_time': run_time.to_numpy(), 'run_spd': run_spd})
    D['saccades'] = detect_saccades(D['eye_data'], vel_thresh=30, r2_thresh=0.8)

    stimsets = [s for s in session.stimulus_names if 'drifting_gratings' in s]

    D['run_epochs'] = get_run_epochs(D['run_data']['run_time'], D['run_data']['run_spd'])
    
    # v1 units
    D['units'] = session.units #[session.units["ecephys_structure_acronym"] == 'VISp']
    D['spikes'] = session.spike_times
    presentations = [session.get_stimulus_table(s) for s in stimsets]
    D['gratings'] = pd.concat( presentations, axis=0)

    return D

def preprocess_gaze_data(gazedata):
    """
        Process gaze data from an allen brain observatory gazedata dataframe

        returns dict 
        -------
            eye_time : np.array of timestamp for samples (upsampled to 1kHz)
            eye_x : np.array of x gaze position
            eye_y : np.array of y gaze position
            pupil : np.array of pupil size
            fs_orig: original sampling rate
            fs_new: new sampling rate
    """
    eye_time = gazedata.index.values
    fs_orig = 1/np.median(np.diff(eye_time))

    eye_x = gazedata.raw_screen_coordinates_spherical_x_deg.interpolate(method='spline', order=5).values
    eye_y = gazedata.raw_screen_coordinates_spherical_y_deg.interpolate(method='spline', order=5).values
    pupil = gazedata.raw_pupil_area.values

    exfun = interp1d(eye_time, eye_x, kind='linear')
    eyfun = interp1d(eye_time, eye_y, kind='linear')
    pupfun = interp1d(eye_time, pupil, kind='linear')

    new_eye_time = np.arange(eye_time[0], eye_time[-1], 0.001)
    eye_x = exfun(new_eye_time)
    eye_y = eyfun(new_eye_time)
    pupil = pupfun(new_eye_time)
    eye_time = new_eye_time
    fs = 1/np.median(np.diff(eye_time))

    out = struct()
    out['eye_time'] = eye_time
    out['eye_x'] = eye_x
    out['eye_y'] = eye_y
    out['pupil'] = pupil
    out['fs_orig'] = fs_orig
    out['fs'] = fs

    return out

def find_zero_crossings(x, mode=0):
    """
        Finds the zero crossings of a signal (x)
        mode = 0: returns all indices of the zero crossings
        mode = 1: returns positive going zero crossings
        mode = -1: returns negative going zero crossings
    """
    inds = np.where(x)
    dat = x[inds[0]]
    if mode == 0:
        return inds[0][np.where(np.abs(np.diff(np.sign(dat)))==2)[0]]
    elif mode == 1:
        return inds[0][np.where(np.diff(np.sign(dat))==2)[0]]
    elif mode == -1:
        return inds[0][np.where(np.diff(np.sign(dat))==-2)[0]]

def fit_gauss_trick(y):
    """
        Fits a gaussian to a signal (y) using a hack
        Instead of fitting a proper gaussian, this function fits a 2nd order polynomial
        to the log of the data (adjusted to be positive). This is much faster (and less accurate)
        than proper least squares gaussian fitting, but it gives a good-enough estimate when we
        have to fit thousands of gaussians quickly
    """
    x = np.arange(0, len(y)) - np.argmax(y)
    y = y - np.min(y) + 1e-3
    iivalid = np.where(y > .5*np.max(y))[0]
    ly = np.log(y)
    X = np.concatenate( [x[:,None] , x[:,None]**2, np.ones( (len(x), 1))], axis=1)
    wts = np.linalg.solve(X[iivalid,:].T@X[iivalid,:], X[iivalid,:].T@ly[iivalid])
    
    return y, np.exp(X@wts)

def r_squared(y, yhat):
    return 1 - np.sum((y - yhat)**2) / np.sum((y - np.mean(y))**2)


def detect_saccades(eye_data, vel_thresh=30,
    accel_thresh=5000, r2_thresh=.85, filter_length=101,
    debug=False, win=100):
    """
        Saccade detection algorithm tailored for mouse data
        Input:
            eye_data: dict of eye data (see preprocess_gaze_data)
            vel_thresh: threshold for velocity (deg/s)
            accel_thresh: threshold for acceleration (deg/s^2)
            r2_thresh: threshold for r-squared of gaussian fit to spd profile of each saccade
            filter_length: length of savitzy-golay differentiating filter
            debug: if True, plots the saccade detection results (default: False)
        Output:
            saccades: dict of saccade data
                saccades['start_time']: np.array of start times of saccades (in seconds)
                saccades['end_time']: np.array of end times of saccades (in seconds)
                saccades['peak_time']: np.array of peak times of saccades (in seconds)
                saccades['start_index']: np.array of indices of start times of saccades (in samples)
                saccades['stop_index']: np.array of indices of end times of saccades (in samples)
                saccades['peak_index']: np.array of indices of peak times of saccades (in samples)
                saccades['peak_vel']: np.array of peak velocities of saccades (in deg/s)
                saccades['dx']: np.array of x-displacements of saccades (in deg)
                saccades['dy']: np.array of y-displacements of saccades (in deg)
                saccades['r2']: np.array of r-squared of gaussian fit to spd profile of each saccade
                saccades['dist_traveled']: np.array of total distance traveled during padded saccade window (used for debugging)
                
    """

    fs = eye_data['fs'].astype(int)
    fs_orig = eye_data['fs_orig'].astype(int)
    offset = win * 2
    flength = filter_length
    grpdelay = (flength-1)/2
    velx = savgol_filter(eye_data['eye_x'], window_length=flength, polyorder=5, deriv=1, delta=1)*fs
    vely = savgol_filter(eye_data['eye_y'], window_length=flength, polyorder=5, deriv=1, delta=1)*fs

    spd = np.hypot(velx, vely)
    accel = savgol_filter(spd, window_length=flength, polyorder=5, deriv=1, delta=1)*fs

    # discretize acceleration based on threshold and find zero crossings
    decimated = np.round(accel/accel_thresh)*accel_thresh
    # this gives us velocity peaks
    zc = find_zero_crossings(decimated, mode=-1)


    if debug:
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(eye_data['eye_time'], eye_data['eye_y'])
        plt.ylim((-5,5))


        plt.subplot(3,1,2)
        plt.plot(eye_data['eye_time'], spd)
        plt.ylim((0, 300))

        plt.subplot(3,1,3)
        plt.plot(eye_data['eye_time'], accel)
        
        plt.plot(eye_data['eye_time'], decimated)

        plt.plot(eye_data['eye_time'][zc], decimated[zc], 'o')
        plt.ylim((0,10000))

    pot_saccades = zc

    print("Found %d potential saccade" %len(pot_saccades))

    pot_saccades = pot_saccades[spd[pot_saccades] > vel_thresh]
    nsac = len(pot_saccades)
    print("%d saccades exceed velocity threshold" %nsac)
    saccades = np.zeros((nsac, 11))

    if debug:
        plt.figure()

    # [tstart, tstop, tpeak, istart, istop, ipeak, dx, dy, peakvel]
    for ii in range(nsac):
        try:
            t0 = np.max( (zc[ii] - win, 0))
            t1 = np.min( (zc[ii] + win, len(eye_data['eye_time'])))
            xax = eye_data['eye_time'][t0:t1] - eye_data['eye_time'][zc[ii]]
            zcs = find_zero_crossings(spd[t0:t1]-vel_thresh, mode=0)
            ttimes = xax[zcs]
            iipos = ttimes < 0
            if np.sum(iipos) > 0:
                istart = zcs[ttimes==np.max(ttimes[iipos])][0] + t0
            else:
                istart = zcs[ttimes==0]-15

            iineg = ttimes > 0
            if np.sum(iineg) > 0:
                istop = zcs[ttimes==np.min(ttimes[iineg])][0] + t0
            else:
                istop = zcs[ttimes==0]+15

            tstart = eye_data['eye_time'][istart]
            tstop = eye_data['eye_time'][istop]

            saccades[ii,0] = tstart
            saccades[ii,1] = tstop
            saccades[ii,3] = istart
            saccades[ii,4] = istop

            dx = eye_data['eye_x'][istop] - eye_data['eye_x'][istart]
            dy = eye_data['eye_y'][istop] - eye_data['eye_y'][istart]

            dpre = np.sqrt((np.mean(eye_data['eye_x'][istop:t1]) - np.mean(eye_data['eye_x'][t0:istart]))**2 + (np.mean(eye_data['eye_y'][istop:t1]) - np.mean(eye_data['eye_y'][t0:istart]))**2)

            peakvel = spd[zc[ii]]

            # saccade velocity QA
            # y = spd[t0:t1]
            # y = spd[istart:istop]
            
            y = spd[ (istart-2*offset):(istop+2*offset)]
            nans = np.isnan(y)
            y,yhat = fit_gauss_trick(y[~nans])
            r2 = r_squared(y, yhat)

            saccades[ii,:] = [tstart, tstop, eye_data['eye_time'][zc[ii]], istart, istop, zc[ii], dx, dy, peakvel, dpre, r2]

            if debug:
                plt.subplot(3,1,1)
                plt.plot(xax, eye_data['eye_x'][t0:t1], '-o')
                plt.subplot(3,1,2)
                plt.plot(xax, eye_data['eye_y'][t0:t1], '-o')
                plt.subplot(3,1,3)
                plt.plot(xax, spd[t0:t1]-vel_thresh, '-o')
                plt.axhline(0, color='k')

                zcs = find_zero_crossings(spd[t0:t1]-vel_thresh, mode=0)
                istart = zcs[0] + t0
                istop = zcs[-1] + t0

                plt.plot(xax[zcs], spd[t0+zcs]-vel_thresh, 'o')
                ttimes = xax[zcs]
                # istart = zcs[ttimes==np.max(ttimes[ttimes < 0])] + t0
                # istop = zcs[ttimes==np.min(ttimes[ttimes > 0])] + t0
                tstart = eye_data['eye_time'][istart]
                tstop = eye_data['eye_time'][istop]

                plt.plot(tstart-eye_data['eye_time'][zc[ii]], 0, 'o')
                plt.plot(tstop-eye_data['eye_time'][zc[ii]], 0, 'o')

        except:
            pass

    
    saccades = saccades[np.where(np.sum(saccades,axis=1)!=0)[0],:]
    print("Found %d saccades" %len(saccades))
    saccades = saccades[np.where(saccades[:,-1]>r2_thresh)[0],:]
    print("Found %d saccades > r-squared threshold" %len(saccades))
    valid_saccades = np.where((saccades[1:,0]-saccades[:-1,1]) > 0.01)[0]+1
    saccades = saccades[valid_saccades,:]
    print("Found %d saccades that don't have overlapping times" %len(saccades))
    print("%d Total saccades" %len(saccades))
    dt = saccades[-1,0]-saccades[0,0]
    print("comes to %02.2f saccades / sec" %(len(saccades)/dt))

    out = {'start_time': saccades[:,0],
        'stop_time': saccades[:,1],
        'peak_time': saccades[:,2],
        'start_index': saccades[:,3],
        'stop_index': saccades[:,4],
        'peak_index': saccades[:,5],
        'dx': saccades[:,6], 'dy': saccades[:,7],
        'peak_vel': saccades[:,8],
        'dist_traveled': saccades[:,9], 'r2': saccades[:,10]}

    return out

def get_run_epochs(run_time, run_spd,
    refrac = 1.0, thresh = 3,
    win = 100, debug=False):

    from scipy.signal import savgol_filter

    run_sm = savgol_filter(run_spd, 31, 3)

    isrunning = (run_sm > thresh).astype(float)

    isrunning[0] = 0
    isrunning[-1] = 0

    startstops = np.diff( isrunning )
    starts = np.where(startstops > 0)[0]
    stops = np.where(startstops < 0)[0]

    print("Found %d potential running epochs" %len(starts))

    # remove starts and stops that are too close together
    dt = np.median(np.diff(run_time))
    bad = np.where(starts[1:] - stops[:-1] < refrac//dt)[0]
    print("Removing %d bad epochs" %len(bad))
    starts = np.delete(starts, bad+1)
    stops = np.delete(stops, bad)

    if debug:
        plt.figure()
        plt.plot(run_time, run_sm)
        for i in range(len(starts)):
            plt.axvline(run_time[starts[i]], color='r')

    npot = len(starts)
    print("Found %d potential running epochs" %npot)
    run_epochs = struct({'start_time': [], 'stop_time': [], 'start_index': [], 'stop_index': []})

    # plt.figure()
    for i in range(npot):

        idx = np.arange(np.maximum(-win + starts[i], 0), np.minimum(stops[i] + win, len(run_spd)), 1) 

        # plt.plot(idx, run_sm[idx], 'k')

        zcstarts = find_zero_crossings(run_sm[idx], mode=1)
        if len(zcstarts)==0 or np.sum(zcstarts <= win) == 0:
            zcstarts = win
        else:
            zcstarts = np.max(zcstarts[zcstarts <= win])

        zcstops = find_zero_crossings(run_sm[idx], mode=-1)
        if len(zcstops)==0 or np.sum(zcstops > (len(idx) - win)) == 0:
            zcstops = len(idx)-win
        else:    
            zcstops = np.min(zcstops[zcstops > (len(idx) - win)])


        # plt.plot(idx, np.round(run_sm[idx]), 'b')

        # plt.plot(idx[zcstarts], run_sm[idx[zcstarts]], 'ro')
        # plt.plot(idx[zcstops], run_sm[idx[zcstops]], 'go')
        # plt.xlim((idx[0], idx[-1]))
        start_idx = np.max( (1, idx[zcstarts]))
        stop_idx = np.min( (len(run_time), idx[zcstops]))
        run_epochs['start_time'].append(run_time[start_idx])
        run_epochs['stop_time'].append(run_time[stop_idx])
        run_epochs['start_index'].append(start_idx)
        run_epochs['stop_index'].append(stop_idx)

    run_epochs['start_time'] = np.array(run_epochs['start_time'])
    run_epochs['stop_time'] = np.array(run_epochs['stop_time'])
    run_epochs['start_index'] = np.array(run_epochs['start_index'])
    run_epochs['stop_index'] = np.array(run_epochs['stop_index'])


    dt = run_epochs['start_time'][1:] - run_epochs['stop_time'][:-1]

    bad = np.where(dt < refrac)[0]
    good = np.setdiff1d(np.arange(len(dt)), bad)
    run_epochs['stop_time'][bad-1] = run_epochs['stop_time'][bad]
    run_epochs['stop_index'][bad-1] = run_epochs['stop_index'][bad]

    for key in run_epochs.keys():
        run_epochs[key] = run_epochs[key][good]

    duration = run_epochs['stop_time'] - run_epochs['start_time']
    bad = np.where(duration < refrac)[0]
    for key in run_epochs.keys():
        run_epochs[key] = np.delete(run_epochs[key], bad)

    return run_epochs

def psth_interp(time, data, onsets, time_bins):
    NT = len(time_bins)
    NStim = len(onsets)
    rspd = np.zeros((NT, NStim))
    fint = interp1d(time, data, kind='linear')
    t0 = time[0]
    t1 = time[-1]

    for istim in range(NStim):
        time_sample = time_bins + onsets[istim]
        idx = np.where(np.logical_and(time_sample >= t0, time_sample <= t1))[0]
        rspd[idx, istim] = fint(time_sample[idx])
    
    return rspd

# import interp1d from scipy
def psth(spike_times, onsets, time_bins):
    bin_size = np.mean(np.diff(time_bins))
    NC = len(spike_times)
    NStim = len(onsets)
    NT = len(time_bins)
    Robs = np.zeros((NT, NStim, NC))
    for istim in range(NStim):
        for iunit in range(NC):
            st = spike_times[iunit]-onsets[istim]
            st = st[np.logical_and(st >= time_bins[0], st <= (time_bins[-1]+bin_size))]
            Robs[np.digitize(st, time_bins)-1,istim, iunit] += 1

    return Robs

def get_valid_time_idx(times, valid_start, valid_stop):
    valid = np.zeros(len(times), dtype=bool)
    for t in range(len(valid_start)):
        valid = np.logical_or(valid, np.logical_and(times >= valid_start[t], times <= valid_stop[t]))
    return valid

def make_animation(D):
    #%% Make animation of it
    import matplotlib.pyplot as plt
    from matplotlib import animation

    onsets = D.gratings.start_time.to_numpy()

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(xlim=(0, 10), ylim=(-2, 2))
    igrat = 0

    ax1 = plt.subplot(3,1,1)
    plt.plot(D.eye_data['eye_time'], D.eye_data['eye_y'], color='k')

    nsac = len(D.saccades['start_time'])
    for ii in range(nsac):
        iix = np.arange(D.saccades['start_index'][ii], D.saccades['stop_index'][ii], 1, dtype=int)
        plt.plot(D.eye_data['eye_time'][iix], D.eye_data['eye_y'][iix], color='r')

    ax1.set_xlim( (-1 + onsets[igrat], onsets[igrat] + 60))
    ax1.set_ylim( (-10, 10))
    ax1.set_ylabel('Eye Y (deg)')

    ax2 = plt.subplot(3,1,2)
    plt.plot(D.eye_data['eye_time'], D.eye_data['pupil'], color='k')
    for i in range(len(D.saccades['start_time'])):
        plt.axvline(D.saccades['start_time'][i], color='r', linestyle='-')
    ax2.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
    ax2.set_ylim((0, .005))
    ax2.set_ylabel('Pupil Area')

    ax3 = plt.subplot(3,1,3)
    line = plt.plot(D.run_data['run_time'], D.run_data['run_spd'], color='k')[0]
    for i in range(len(D.run_epochs['start_time'])):
        plt.axvspan(D.run_epochs['start_time'][i], D.run_epochs['stop_time'][i], color='gray', alpha=.5)
    for i in range(len(D.saccades['start_time'])):
        plt.axvline(D.saccades['start_time'][i], color='r', linestyle='-')
    ax3.set_xlim( (0 + onsets[igrat], onsets[igrat] + 60))
    ax3.set_ylim((0, 55))
    ax3.set_xlabel('Seconds')
    ax3.set_ylabel('Speed (cm/s)')
    # line, = ax.plot([], [], lw=2)

    def init():
        line.set_data(D.run_data['run_time'], D.run_data['run_spd'])
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        ax1.set_xlim( (0 + onsets[i], onsets[i] + 60))
        ax2.set_xlim( (0 + onsets[i], onsets[i] + 60))
        ax3.set_xlim( (0 + onsets[i], onsets[i] + 60))
        line.set_data(D.run_data['run_time'], D.run_data['run_spd'])
        return line,
        
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=range(np.minimum(400, len(onsets))), interval=1, blit=True, repeat=False)

    return anim

def nanmean(x):
    return np.mean(x*~np.isnan(x), axis=0)


def save_analyses(D, fpath='./analyses/'):
    import pickle
    fname = os.path.join(fpath, D.files.session + '.pkl')
    # save the data to a file with pickle
    with open(fname, 'wb') as f:
        pickle.dump(D, f)

def load_analyses(session_name, fpath='./analyses/'):
    import pickle
    fname = os.path.join(fpath, session_name + '.pkl')
    D = pickle.load(open(fname, 'rb'))
    return D

# TODO: oof, this is bad form. Clean these up ASAP
def bootstrap_ci(data, n = None, nboot=500):
    if n is None:
        n = data.shape[1]
    bootix = np.random.randint(0,n-1,size=(nboot,n))
    return np.percentile(np.nanmean(data[:,bootix,:], axis=2), (15.75, 84.25), axis=1)

def boot_ci(data, n = None, nboot=500):
    if n is None:
        n = data.shape[1]
    bootix = np.random.randint(0,n-1,size=(nboot,n))
    return np.percentile(np.nanmean(data[bootix,:], axis=1), (2.5, 50, 97.5), axis=0)

def main_analysis(D, hack_valid_run=False):
    D['psths'] = struct({'saccade_onset': struct(),
    'run_onset': struct(),
    'grat_onset': struct(),
    'grat_tuning': struct()})

    #%% plot average running speed aligned to saccade onset
    from scipy.signal import savgol_filter

    time_bins = np.arange(-4, 4, .01)

    run_time = D.run_data.run_time
    run_spd = np.maximum(D.run_data.run_spd, 0)
    eye_time = D.eye_data.eye_time
    pupil = D.eye_data.pupil


    sac_onsets = D.saccades['start_time']
    sac_onsets = sac_onsets[np.logical_and(sac_onsets > run_time[0] + .5, sac_onsets < run_time[-1] - .5)]
    breaks = np.where(np.diff(sac_onsets)>5)[0]

    plt.figure()
    plt.plot(sac_onsets, np.ones(len(sac_onsets)), 'o')
    val_ix = np.ones(len(run_time), dtype=bool)
    for i in range(len(breaks)):
        plt.axvline(sac_onsets[breaks[i]], color='k', linestyle='--')
        plt.axvline(sac_onsets[breaks[i]+1], color='r', linestyle='--')
        ix = np.logical_and(run_time >= sac_onsets[breaks[i]], run_time <= sac_onsets[breaks[i]+1])
        val_ix[ix] = False

    if hack_valid_run:
        good = val_ix #
    else:
        good = ~np.isnan(run_time)

    plt.plot(run_time, run_spd/np.max(run_spd), 'k')
    plt.plot(run_time, good)

    rspd = psth_interp(run_time[good], run_spd[good], sac_onsets, time_bins)
    m = np.nanmean(rspd, axis=1)
    sd = np.nanstd(rspd, axis=1) / np.sqrt(rspd.shape[1])
    plt.figure()

    f = plt.fill_between(time_bins, m-sd*2, m+sd*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    # val_ix = D.run_data.run_time > 0
    plt.axhline(np.mean(run_spd[good]), color='k', linestyle='--')
    plt.xlabel('Saccade Onset (sec)')
    plt.ylabel('Running Speed (cm/s)')

    D.psths.saccade_onset['run_spd'] = struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})

    #%% plot histogram of saccade onsets around running onset
    run_onsets = D.run_epochs.start_time
    time_bins = np.arange(-1, 2.5, .02)
    sachist = psth([D.saccades['start_time']], run_onsets, time_bins)
    plt.figure()
    plt.fill_between(time_bins, np.zeros(len(time_bins)), np.sum(sachist, axis=1).flatten(), color='k', alpha=.5)
    plt.plot(time_bins, np.sum(sachist, axis=1), color='k')
    plt.xlabel("Time from Running Onset (s)")
    plt.ylabel("Saccade Count")

    D.psths.run_onset['saccade_onset'] = struct({'hist': np.sum(sachist, axis=1).flatten(), 'time_bins': time_bins})

    #%% plot pupil area aligned to saccade onset and running onset

    pupsac = psth_interp(eye_time, pupil, sac_onsets, time_bins)
    puprun = psth_interp(eye_time, pupil, run_onsets, time_bins)
    runrun = psth_interp(run_time[good], run_spd[good], run_onsets, time_bins)

    # pupil area aligned to saccade onset
    plt.figure()
    plt.subplot(1,2,1)
    m = np.nanmean(pupsac, axis=1)
    s = np.nanstd(pupsac, axis=1) / np.sqrt(pupsac.shape[1])
    D.psths.saccade_onset['pupil'] = struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Saccade Onset (s)')
    plt.ylabel('Pupil Area (?)')

    plt.subplot(1,2,2)
    m = np.nanmean(puprun, axis=1)
    s = np.nanstd(puprun, axis=1) / np.sqrt(puprun.shape[1])
    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Running Onset (s)')
    plt.ylabel('Pupil Area (?)')

    D.psths.run_onset['pupil'] = struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    #%% control: running speed as a function of runing onset
    plt.figure()

    m = np.nanmean(runrun, axis=1)
    s = np.nanstd(runrun, axis=1) / np.sqrt(runrun.shape[1])
    plt.fill_between(time_bins, m-s*2, m+s*2, color='k', alpha=.5)
    plt.plot(time_bins, m, 'k')
    plt.xlabel('Time from Running Onset (s)')
    plt.ylabel('Running Speed (cm/s)')

    D.psths.run_onset['run_spd'] = struct({'mean': m, 'std_error': s, 'time_bins': time_bins})

    #%% Analyze spikes!
    from scipy.signal import savgol_filter

    idx = get_valid_time_idx(D.saccades['start_time'], D.gratings['start_time'].to_numpy(), D.gratings['stop_time'].to_numpy())

    units = D.units[D.units["ecephys_structure_acronym"] == 'VISp']
    # units = D.units
    spike_times = [D.spikes[id] for id in units.index.values]

    bin_size = 0.001
    time_bins = np.arange(-.5, .5, bin_size)


    plt.figure(figsize=(10,3))
    Robs = psth(spike_times, D.gratings['start_time'].to_numpy(), time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,1)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Grating")
    plt.ylabel("Firing Rate (sp / s)")
    plt.xlabel("Time from Onset (s)")

    D.psths.grat_onset['spikes'] = struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})


    Robs = psth(spike_times, D.saccades['start_time'], time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,2)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Saccade")
    plt.xlabel("Time from Onset (s)")

    D.psths.saccade_onset['spikes'] = struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})

    Robs = psth(spike_times, D.run_epochs.start_time, time_bins)
    m = np.mean(Robs, axis=1).squeeze()/bin_size
    sd = np.std(Robs, axis=1).squeeze()/bin_size / np.sqrt(Robs.shape[1])
    plt.subplot(1,3,3)
    f = plt.plot(time_bins, savgol_filter(m, 101, 3, axis=0))
    plt.title("Running")
    plt.xlabel("Time from Onset (s)")

    D.psths.run_onset['spikes'] = struct({'mean': m, 'std_error': sd, 'time_bins': time_bins})


    #%% PSTH / Tuning Curve analysis
    # %matplotlib ipympl

    bin_size = .02
    t1 = np.max(D.gratings.duration )
    time_bins = np.arange(-.25, t1+.25, bin_size)
    Robs = psth(spike_times, D.gratings['start_time'].to_numpy(), time_bins)
    run_spd  = psth_interp(D.run_data.run_time, D.run_data.run_spd, D.gratings['start_time'].to_numpy(), time_bins)
    pupil  = psth_interp(D.eye_data.eye_time, D.eye_data.pupil, D.gratings['start_time'].to_numpy(), time_bins)
    saccades = psth([D.saccades['start_time']], D.gratings['start_time'].to_numpy(), time_bins).squeeze()

    eye_dx = savgol_filter(D.eye_data.eye_x, 101, 3, axis=0, deriv=1, delta=1/D.eye_data.fs)
    eye_dy = savgol_filter(D.eye_data.eye_y, 101, 3, axis=0, deriv=1, delta=1/D.eye_data.fs)
    eye_spd = np.hypot(eye_dx, eye_dy)
    eye_vel = psth_interp(D.eye_data.eye_time, eye_spd, D.gratings['start_time'].to_numpy(), time_bins)

    wrapAt = 180
    bad_ix = (D.gratings.orientation=='null')
    condition = D.gratings.orientation
    condition[bad_ix] = 0.1
    condition = condition % wrapAt
    conds = np.unique(condition)
    Nconds = len(conds)

    #%%
    plt.figure(figsize=(10,3))
    plt.subplot(1,4,1)
    plt.imshow(run_spd.T, aspect='auto')
    plt.subplot(1,4,2)
    plt.imshow(pupil.T, aspect='auto')
    plt.subplot(1,4,3)
    plt.imshow(eye_vel.T, aspect='auto')

    # main boot analysis
    run_mod = dict()

    stationary_trials = np.where(nanmean(run_spd) < 3)[0]
    num_stat = len(stationary_trials)
    running_trials = np.where(nanmean(run_spd) > 5)[0]
    num_run = len(running_trials)

    num_trials = np.minimum(num_stat, num_run)
    print("using %d trials [%d run, %d stationary]" % (num_trials, num_run, num_stat))

    tstim = np.logical_and(time_bins > 0.05, time_bins < t1 + 0.05)
    tbase = time_bins < 0

    spctrun = Robs[:, running_trials, :]
    spctstat = Robs[:, stationary_trials, :]

    frbaseR = boot_ci(np.mean(spctrun[tbase,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frbaseS = boot_ci(np.mean(spctstat[tbase,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frstimR = boot_ci(np.mean(spctrun[tstim,:,:], axis=0), n = num_trials, nboot=500)/bin_size
    frstimS = boot_ci(np.mean(spctstat[tstim,:,:], axis=0), n = num_trials, nboot=500)/bin_size

    plt.figure()
    plt.subplot(1,2,1)
    plt.errorbar(frbaseS[1,:], frbaseR[1,:], yerr=np.abs(frbaseR[(0,2),:] - frbaseR[1,:]),
        xerr=np.abs(frbaseS[(0,2),:] - frbaseS[1,:]), fmt='o')
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel('Stationary')
    plt.ylabel('Running')

    plt.subplot(1,2,2)
    plt.errorbar(frstimS[1,:], frstimR[1,:], yerr=np.abs(frstimR[(0,2),:] - frstimR[1,:]),
        xerr=np.abs(frstimS[(0,2),:] - frstimS[1,:]), fmt='o')
    plt.plot(plt.xlim(), plt.xlim(), 'k--')
    plt.xlabel('Stationary')
    plt.ylabel('Running')

    D['firing_rate'] = struct({'base_rate_stat': frbaseS, 'base_rate_run': frbaseR,
        'stim_rate_run': frstimR, 'stim_rate_stat': frstimS})

    #%% analyze tuning curves

    nboot = 500


    NC = Robs.shape[-1]
    mu_rate = struct({'all':np.zeros((Nconds, len(time_bins), NC)),
        'running': np.zeros((Nconds, len(time_bins), NC)),
        'stationary': np.zeros((Nconds, len(time_bins), NC)),
        'running_ns': np.zeros((Nconds, len(time_bins), NC)),
        'stationary_ns': np.zeros((Nconds, len(time_bins), NC)),
        'pupil_large': np.zeros((Nconds, len(time_bins), NC)),
        'pupil_small': np.zeros((Nconds, len(time_bins), NC))})

    n = dict()
    ci_rate = dict()
    for key in mu_rate.keys():
        n[key] = np.zeros(Nconds)
        ci_rate[key] = np.zeros((Nconds, 2, len(time_bins), NC))


    for icond, cond in enumerate(conds):
        
        # all trials
        idx = np.logical_and(condition == cond, D.gratings.duration > .6)
        idx = np.where(idx)[0]
        mu_rate['all'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['all'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['all'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) > 5)
        mu_rate['running'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['running'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['running'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) < 3)
        mu_rate['stationary'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['stationary'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['stationary'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(pupil, axis=0) > np.nanmedian(pupil))
        mu_rate['pupil_large'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['pupil_large'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['pupil_large'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(pupil, axis=0) < np.nanmedian(pupil))
        mu_rate['pupil_small'][icond, :, :] = np.nanmean(Robs[:, idx, :], axis=1)
        ci_rate['pupil_small'][icond,:] = bootstrap_ci(Robs[:, idx, :], nboot=nboot)
        n['pupil_small'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.nanmean(run_spd, axis=0) > 5)
        # stat_time = np.expand_dims(eye_vel[:,idx]<25, axis=2)
        # mu_rate['running_ns'][icond, :, :] = np.mean(Robs[:, idx, :]*stat_time, axis=1)
        
        # idx = np.logical_and(idx, np.sum(saccades, axis=0) == 0)
        idx = np.logical_and(idx, np.sum(eye_vel > 50, axis=0) == 0)
        mu_rate['running_ns'][icond, :, :] = np.mean(Robs[:, idx, :], axis=1)
        n['running_ns'][icond] = np.sum(idx)

        idx = np.logical_and(condition == cond, np.mean(run_spd, axis=0) < 3)
        # stat_time = np.expand_dims(eye_vel[:,idx]<25, axis=2)
        # mu_rate['stationary_ns'][icond, :, :] = np.mean(Robs[:, idx, :]*stat_time, axis=1)

        idx = np.logical_and(idx, np.sum(eye_vel > 50, axis=0) == 0)
        mu_rate['stationary_ns'][icond, :, :] = np.mean(Robs[:, idx, :], axis=1)

        n['stationary_ns'][icond] = np.sum(idx)
        
    D.psths.grat_tuning = struct({'mu_rate': mu_rate, 'n': n, 'ci_rate': ci_rate, 'time_bins': time_bins})
    return D

# utility for checking size of a dict
from __future__ import print_function
from sys import getsizeof, stderr
from itertools import chain
from collections import deque
try:
    from reprlib import repr
except ImportError:
    pass

def total_size(o, handlers={}, verbose=False):
    """ Returns the approximate memory footprint an object and all of its contents.

    Automatically finds the contents of the following builtin containers and
    their subclasses:  tuple, list, deque, dict, set and frozenset.
    To search other containers, add handlers to iterate over their contents:

        handlers = {SomeContainerClass: iter,
                    OtherContainerClass: OtherContainerClass.get_elements}

    """
    dict_handler = lambda d: chain.from_iterable(d.items())
    all_handlers = {tuple: iter,
                    list: iter,
                    deque: iter,
                    dict: dict_handler,
                    set: iter,
                    frozenset: iter,
                   }
    all_handlers.update(handlers)     # user handlers take precedence
    seen = set()                      # track which object id's have already been seen
    default_size = getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = getsizeof(o, default_size)

        if verbose:
            print(s, type(o), repr(o), file=stderr)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)