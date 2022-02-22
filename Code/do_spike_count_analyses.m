function Stat = do_spike_count_analyses(D, opts)
% Stat = do_spike_count_analyses(D, opts)
% 
% defopts.winstart = 0.035;
% defopts.prewin = .2;
% defopts.postwin = .1;
% defopts.binsize = .01;
% defopts.run_thresh = 3;
% defopts.debug = false;
% defopts.nboot = 100;
% defopts.spike_rate_thresh = 1;

rng(1234) % for reproducibility

if nargin < 2
    opts = struct();
end

defopts = struct();
defopts.winstart = 0.035;
defopts.prewin = .2;
defopts.postwin = .1;
defopts.binsize = .01;
defopts.run_thresh = 3;
defopts.debug = false;
defopts.nboot = 100;
defopts.spike_rate_thresh = 1;
defopts.baseline_subtract = true; % subtract baseline before computing OSI
defopts.weighted_spike_count = false;

opts = mergeStruct(defopts, opts);

cids = unique(D.spikeIds);
NC = numel(cids);


Stat = struct();
% reproduce some the metrics from the Allen Institute analyses
Stat.pvis = nan(NC, 3); % p-value from anova1 (spikes vs condition), where condition is either all or direction only
Stat.cid = nan(NC, 1);  % unit id
Stat.nori = nan(NC,3);  % number of trials per orientation
Stat.ndir = nan(NC, 3); % number of trials per direction
Stat.runmod = nan(NC,2); % running modulation (Allen institute metric)
Stat.murun = nan(NC,2); % mean running (at pref stimulus)
Stat.mustat = nan(NC,2); % mean not running (at pref stimulus)
Stat.prun = nan(NC,2); % pvalue for running (Allen Institute metric)

% bootstrapped firing rates under different conditions
Stat.frBaseR = nan(NC, 3); % baseline firing rate during running (errorbars and mean)
Stat.frBaseS = nan(NC, 3);
Stat.frStimR = nan(NC, 3); 
Stat.frStimS = nan(NC, 3);
Stat.frPrefR = nan(NC, 3);
Stat.frPrefS = nan(NC, 3);

% Tuning curves 
total_directions = unique(D.GratingDirections(~isnan(D.GratingDirections)));
num_directions = numel(total_directions);

Stat.meanrate = nan(NC,1); % mean firing rate
Stat.baselinerate = nan(NC,1);

Stat.directions = total_directions;

Stat.dratemarg = nan(NC,num_directions, 3); % direction tuning curve marginalized over TF and SF
Stat.dratemargR = nan(NC,num_directions, 3); % direction tuning curve marginalized over TF and SF
Stat.dratemargS = nan(NC,num_directions, 3); % direction tuning curve marginalized over TF and SF

Stat.dratebest = nan(NC,num_directions, 3); % direction tuning curve at best SF
Stat.dratebestR = nan(NC,num_directions, 3); % direction tuning curve at best SF
Stat.dratebestS = nan(NC,num_directions, 3); % direction tuning curve at best SF

Stat.drateweight = nan(NC,num_directions, 3); % direction tuning curve weighted by TF / SF tuning
Stat.drateweightR = nan(NC,num_directions, 3); % direction tuning curve weighted by TF / SF tuning
Stat.drateweightS = nan(NC,num_directions, 3); % direction tuning curve weighted by TF / SF tuning

Stat.robs = cell(NC, 1);
Stat.runningspeed = cell(NC, 1);

% correlation with running
Stat.runrho = nan(NC,1);
Stat.runrhop = nan(NC,1);

Stat.rfecc = nan(NC,1);
Stat.rfsig = nan(NC,1);

Stat.prctilerunbootmarg = nan(NC, 1);
Stat.prctilerunbootbest = nan(NC, 1);
Stat.prctilerunbootweight = nan(NC, 1);

Stat.bootTestMedianfrStim = nan(NC, 1);
Stat.bootTestMeanfrStim = nan(NC, 1);
Stat.bootTestMedianfrBase = nan(NC, 1);
Stat.bootTestMeanfrBase = nan(NC, 1);

for cc = 1:NC

    fprintf('%d/%d\n', cc, NC)
    cid = cids(cc);
    Stat.cid(cc) = cid;
    
    try
        if ~isempty(D.units{cid})
            Stat.rfecc(cc) = hypot(D.units{cid}{1}.center(1), D.units{cid}{1}.center(2));
            Stat.rfsig(cc) = D.units{cid}{1}.maxV > 6;
        end
    end

    unitix = D.spikeIds == cid;
    sessix = unique(D.sessNumSpikes(unitix));

    gtix = find(ismember(D.sessNumGratings, sessix));
    gtix(isnan(D.GratingDirections(gtix))) = [];

    onsets = D.GratingOnsets(gtix);
    winsize = mode(D.GratingOffsets(gtix) - D.GratingOnsets(gtix));

    t0 = min(onsets) - 2*winsize;
    st = D.spikeTimes(unitix) - t0;
    onsets = onsets - t0;

    st(st < min(onsets)) = [];
    sp = struct('st', st, 'clu', ones(numel(st),1));

    % quick bin spikes during gratings
    R = binNeuronSpikeTimesFast(sp, onsets, winsize);
    R = R ./ winsize;

%     SKIP IF SPIKE RATE DURING GRATINGS BELOW THRESHOLD
    if (mean(R) < opts.spike_rate_thresh) || (mean(R==0)>.5) % skip if spike rate is less that 
        continue
    end

    % --- Get binned spikes, stimulus, behavior
    [stimconds, robs, behavior, unitopts] = bin_ssunit(D, cid, 'win', [-opts.prewin opts.postwin], 'plot', false, 'binsize', opts.binsize);
    
    % bin spikes at 50ms and run an anova
    binmat = (unitopts.lags(:) < 0.05:.05:1.05) - (unitopts.lags(:) < 0:.05:1);
    binmat(:,sum(binmat)==0) = [];
    binmat = binmat ./ sum(binmat);
    
    brobs = robs*binmat;
    
    group = reshape(repmat(stimconds{1}, 1, size(brobs,2)), [], 1);
    Stat.pvis(cc,3) = anova1(brobs(:), group, 'off');

    tix = unitopts.lags > 0 & unitopts.lags < (unitopts.lags(end) - opts.postwin);
    if opts.weighted_spike_count
        kern = mean(filtfilt(boxcar(5)/5, 1, robs'),2);
        kern = kern - mean(kern(unitopts.lags < 0));
        kern = kern - min(kern);
        kern = kern .* (unitopts.lags > 0 & unitopts.lags <  unitopts.lags(end)-opts.postwin)';
        kern = kern / sum(kern);

        R = robs*kern/unitopts.binsize;
    else
        R = sum(robs(:,tix),2) ./ (sum(tix)*unitopts.binsize);
    end

    tix = unitopts.lags < 0 ;
    frbase = sum(robs(:,tix),2) ./ (sum(tix)*unitopts.binsize);
    
    baseline = min(mean(robs(:,tix))) / unitopts.binsize;
    Stat.baselinerate(cc) = baseline;

    % get conditions
    direction = stimconds{1};
    speed = stimconds{2};
    freq = stimconds{3};

    speeds = unique(speed(:))';
    freqs = unique(freq(:))';
    directions = unique(direction(:))';

    % bin all conditions
    Dmat = direction == directions;
    Fmat = freq == freqs;
    Smat = speed == speeds;

    nd = numel(directions);
    nf = numel(freqs);
    ns = numel(speeds);
    nt = numel(R);
    Xbig = zeros(nt, nd, nf, ns);
    for iis = 1:ns
        for iif = 1:nf
            Xbig(:,:,iif,iis) = Dmat .* Fmat(:,iif) .* Smat(:,iis);
        end
    end


    runspeed = nanmean(behavior{1},2); %#ok<*NANMEAN> 
    
    Stat.meanrate(cc) = mean(R);
    Stat.robs{cc} = R;
    Stat.runningspeed{cc} = runspeed;

    ix = runspeed > 0;
    [rho, pval] = corr(R(ix), runspeed(ix), 'Type', 'Spearman');
    Stat.runrho(cc) = rho;
    Stat.runrhop(cc) = pval;

    statix = abs(runspeed) < opts.run_thresh;
    runix = runspeed > opts.run_thresh;

    Xbig = reshape(Xbig, nt, []);

    % get stimulus selectivity
    [i,group] = find(Xbig>0);
    [~, ind] = sort(i);
    group = group(ind);

    pval = anova1(R, group, 'off');
    Stat.pvis(cc,1) = pval;

    % get direction selectivity
    [i,group] = find(Dmat>0);
    [~, ind] = sort(i);
    group = group(ind);

    pval = anova1(R, group, 'off');
    Stat.pvis(cc,2) = pval;
    
    % TUNING CURVES

    % --- direction marginalize over TF/SF (Running)
    direc_ix = ismember(total_directions, directions);

    % --- tuning curve ignoring running condition
    X = Dmat;
    x = X.*R;
    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);
    Stat.dratemarg(cc,direc_ix,:) = ci';
    
    % bootstrapped null hypothesis test by shuffling running condition
    isgood = runix | statix; % get trials that didn't have negative or dropped treadmill data
    [~, mxid] = max(ci(2,:)); % preferred stimulus condition
    prefix = Dmat(:,mxid)>0 & isgood; % index into the preferred direction

    rpref = R(prefix);
    isrun = runix(prefix);

    [~, bstats] = boot_ttest(rpref(isrun), rpref(~isrun), @mean);
    Stat.prctilerunbootmarg(cc) =  bstats.propgt; % fraction of null differences the true difference is greater than
    
    % --- direction marginalize over TF/SF (Running)
    X = Dmat.*runix;
    x = X.*R;
    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);
    Stat.dratemargR(cc,direc_ix,:) = ci';
    
    % --- direction marginalize over TF/SF (Stationary)
    X = Dmat.*statix;
    x = X.*R;
    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);
    Stat.dratemargS(cc,direc_ix,:) = ci';
    

    % --- running modulation at preferred direction (Allen Institute metric)
    Stat.ndir(cc,1) = sum(Dmat(:,mxid));

    Drun = Dmat(:,mxid).*runix;
    murun = sum(Drun.*R) ./ sum(Drun);
    Dstat = Dmat(:,mxid).*statix;
    mustat = sum(Dstat.*R) ./ sum(Dstat);
    C = sign(murun - mustat);
    rmax = max(murun, mustat);
    rmin = min(murun, mustat);
    Stat.runmod(cc,1) = C * (rmax - rmin) / abs(rmin);
    Stat.murun(cc,1) = murun;
    Stat.mustat(cc,1) = mustat;
    [~, Stat.prun(cc,1)] = ttest2(R(Drun>0), R(Dstat>0));

    % --- direction tuning at best SF / TF
    X = Xbig;
    x = X.*R;

    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);
    mu = ci(2,:);
    Ici = reshape(ci', nd, [], 3);
    I = reshape(mu, nd, []);
    [ii,mxid] = find(I==max(I(:)));
    if numel(mxid) > 1
        [~, jj] = max(sum(I));
        mxid = jj;
        [~, ii] = max(I(:,mxid));
    end

    Stat.dratebest(cc,direc_ix,:) = squeeze(Ici(:,mxid,:));

    % null hypothesis test
    [~, prefid] = max(mu);
    prefix = X(:,prefid)>0 & isgood; % index into the preferred direction

    rpref = R(prefix);
    isrun = runix(prefix);
    if (mean(isrun) > .1)
      [~, bstats] = boot_ttest(rpref(isrun), rpref(~isrun), @mean);
      Stat.prctilerunbootbest(cc) = bstats.propgt;
    end

    X = Xbig.*runix;
    x = X.*R;
    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);

    Ici = reshape(ci', nd, [], 3);
    
    Stat.dratebestR(cc,direc_ix,:) = squeeze(Ici(:,mxid,:));

    X = Xbig.*statix;
    x = X.*R;
    % bootstrap errorbars
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);

    Ici = reshape(ci', nd, [], 3);

    Stat.dratebestS(cc,direc_ix,:) = squeeze(Ici(:,mxid,:));


    % get running modulation at best single stimulus
    [~, id] = max(ci(2,:));
    Drun = Xbig(:,id).*runix;
    murun = sum(Drun.*R) ./ sum(Drun);
    Dstat = Xbig(:,id).*statix;
    mustat = sum(Dstat.*R) ./ sum(Dstat);
    C = sign(murun - mustat);
    rmax = max(murun, mustat);
    rmin = min(murun, mustat);
    Stat.runmod(cc,2) = C * (rmax - rmin) / abs(rmin);
    Stat.murun(cc,2) = murun;
    Stat.mustat(cc,2) = mustat;
    [~, Stat.prun(cc,2)] = ttest2(R(Drun>0), R(Dstat>0));


    n = reshape(sum(Xbig), nd, []);
    
    Stat.ndir(cc,2) = sum(n(:,mxid));
    Stat.ndir(cc,3) = n(ii,mxid); % number of trials at best stimulus

    % --- direction tuning weighted by SF / TF tuning
    w = max(I)'; w = w ./ sum(w);
    
    X = reshape(reshape(Xbig, nt*nd, [])*w, nt, nd);
    x = X.*R;
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);

    Stat.drateweight(cc,direc_ix,:) = ci';
    [~, mxid] = max(ci(2,:));

    % null hypothesis test
    prefix = X(:,mxid)>0 & isgood; % index into the preferred direction

    rpref = R(prefix);
    x = X(prefix, mxid);
    isrun = runix(prefix);
    if mean(isrun) > .1

        npref = numel(rpref);
        bootr = randi(npref, [sum(isrun) opts.nboot]);
        boots = randi(npref, [sum(~isrun) opts.nboot]);

        nulldiff = sum(x(bootr).*rpref(bootr)) ./ sum(x(bootr)) - ...
            sum(x(boots).*rpref(boots)) ./ sum(x(boots));

        truediff = ((x.*isrun)'*rpref) / sum(x.*isrun) - ...
            ((x.*~isrun)'*rpref) / sum(x.*~isrun);

        Stat.prctilerunbootweight(cc) =  mean(truediff > nulldiff); % fraction of null differences the true difference is greater than
    end

    X = reshape(reshape(Xbig, nt*nd, [])*w, nt, nd);
    X = X.*runix;
    x = X.*R;
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);

    Stat.drateweightR(cc,direc_ix,:) = ci';
    
    X = reshape(reshape(Xbig, nt*nd, [])*w, nt, nd);
    X = X.*statix;
    x = X.*R;
    bootix = randi(size(x,1), [size(x,1) opts.nboot]);
    a = squeeze(sum(reshape(x(bootix,:), [size(bootix) size(X,2)]),1)) ./ ...
        squeeze(sum(reshape(X(bootix,:), [size(bootix) size(X,2)]),1));
    ci = prctile(a, [2.5 50 97.5]);

    Stat.drateweightS(cc,direc_ix,:) = ci';

    runTrials = find(runix);
    statTrials = find(statix);

    nrun = numel(runTrials);
    nstat = numel(statTrials);

    n = min(nrun, nstat);
%     n = max(nrun, nstat);

    Stat.frBaseR(cc,:) = prctile(mean(frbase(runTrials(randi(nrun, [n opts.nboot])))), [2.5 50 97.5]);
    Stat.frBaseS(cc,:) = prctile(mean(frbase(statTrials(randi(nstat, [n opts.nboot])))), [2.5 50 97.5]);

    Stat.frStimR(cc,:) = prctile(mean(R(runTrials(randi(nrun, [n opts.nboot])))), [2.5 50 97.5]);
    Stat.frStimS(cc,:) = prctile(mean(R(statTrials(randi(nstat, [n opts.nboot])))), [2.5 50 97.5]);
    
    [~, bstats] = boot_ttest(frbase(runTrials), frbase(statTrials), @mean, opts.nboot);
    Stat.bootTestMeanfrBase(cc) = bstats.propgt;

    [~, bstats] = boot_ttest(R(runTrials), R(statTrials), @mean, opts.nboot);
    Stat.bootTestMeanfrStim(cc) = bstats.propgt;

    [~, bstats] = boot_ttest(frbase(runTrials), frbase(statTrials), @median, opts.nboot);
    Stat.bootTestMedianfrBase(cc) = bstats.propgt;

    [~, bstats] = boot_ttest(R(runTrials), R(statTrials), @median, opts.nboot);
    Stat.bootTestMedianfrStim(cc) = bstats.propgt;
end

Stat.opts = opts;


