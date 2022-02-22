function [S, opts] = running_vs_spikePC(D, sessionId, opts)
% [S, opts] = running_vs_spikePC(D, sessionId, opts)
% plot spiking PC binned by trials -- this way we can avoid smoothing and get rid of nans

defopts = struct();
defopts.prewin = .1;
defopts.postwin = .1;
defopts.spike_rate_thresh = 1;
defopts.ndim = 5;
defopts.plot = false;
defopts.normalization = 'minmax';
defopts.randomize_folds = false;
defopts.folds = 5;

if nargin < 3
    opts = struct();
end

rng(1)

opts = mergeStruct(defopts, opts);

gratix = D.sessNumGratings == sessionId;
treadix = D.sessNumTread == sessionId;
spikeix = D.sessNumSpikes == sessionId;

onsets = D.GratingOnsets(gratix)-opts.prewin;
tstart = min(onsets);
opts.winsize = mode(D.GratingOffsets(gratix) - D.GratingOnsets(gratix)) + opts.prewin + opts.postwin;

% get running speed during grating
treadgood = find(~isnan(D.treadTime) & treadix);
[~, ~, id1] = histcounts(D.GratingOnsets(gratix), D.treadTime(treadgood));
[~, ~, id2] = histcounts(D.GratingOffsets(gratix), D.treadTime(treadgood));
bad = (id1 == 0 | id2 == 0);
runningspeed = nan(numel(D.GratingOnsets(gratix)), 1);
id1 = treadgood(id1(~bad));
id2 = treadgood(id2(~bad));
runningspeed(~bad) = arrayfun(@(x,y) nanmean(D.treadSpeed(x:y)), id1, id2); %#ok<NANMEAN> 

onsets = onsets - tstart;
st = D.spikeTimes(spikeix) - tstart;
clu = D.spikeIds(spikeix);
iix = st > min(onsets); %#ok<*NANMIN> 

cids = unique(clu);
if isfield(D, 'unit_area')
    cids = cids(strcmp(D.unit_area(cids), 'VISp'));
    iix = iix & ismember(clu, cids);
end

r = binNeuronSpikeTimesFast(struct('st', st(iix), 'clu', clu(iix)+1), onsets, opts.winsize);
clist = unique(clu(iix)+1);
r = r(:,clist);
cids = mean(r) ./ opts.winsize > opts.spike_rate_thresh;
r = r(:,cids);

goodix = sum(r,2)~=0 & ~isnan(runningspeed); % trials where at least 1 neuron spiked and runing speed is valid
r = r(goodix,:);
runningspeed = runningspeed(goodix);

NC = size(r,2);
S.cids = clist(cids)-1;

switch opts.normalization
    case 'minmax'
        r = (r - min(r)) ./ range(r);
    case 'zscore'
        r = zscore(r);
end

S.robs = r;

% pca on spikes
[coeff, ~] = pca(r);
coeff = coeff .* sign(sum(sign(coeff))); % sign is irrelevant for PCA, but we care because if most of the units have negative loadings that is meaningful
score = (r-mean(r))*coeff;
% C = cov(r);
% [u,~] = svd(C);
% rpc = r*u(:,1:opts.ndim).*sign(sum(u(:,1:opts.ndim)));
% rpc = r*u(:,1:opts.ndim);
% rpc = (rpc - min(rpc)) ./ range(rpc); % normalize

rpc = score(:,1:opts.ndim);

iix = ~isnan(runningspeed);
[rho, pval] = corr(rpc(iix,:), runningspeed(iix), 'Type', 'Spearman');

nt = numel(runningspeed);


xidxs = regression.xvalidationIdx(nt, opts.folds, opts.randomize_folds);
yhat = nan(nt,1);

what = nan(size(r,2)+1,opts.folds);
for i = 1:opts.folds
    xtrain = r(xidxs{i,1},:);
    ytrain = runningspeed(xidxs{i,1});
    xtest = r(xidxs{i,2},:);
    [what(:,i), ~, outfun] = outputNonlinLs(xtrain,ytrain, struct('display', 'off'));
    yhat(xidxs{i,2}) = outfun(xtest*what(2:end,i) + what(1,i));
end

figure(1); clf
plot(runningspeed, 'k'); hold on
plot(yhat, 'r')

S.pc_coeffs = coeff(:,1:opts.ndim);
S.decoding_r2 = rsquared(runningspeed, yhat);
S.decoding_folds = opts.folds;
S.decoding_beta = what;
S.decoding_fun = outfun;
S.decoding_runspeed = yhat;
S.rpc = rpc;
S.runningspeed = runningspeed;
S.rho = rho;
S.pval = pval;
S.rhounit = nan(NC,1);
S.pvalunit = nan(NC,1);
S.sessionid = sessionId;

for cc = 1:NC
    [rho, pval] = corr(r(iix,cc), runningspeed(iix), 'Type', 'Spearman');
    S.rhounit(cc) = rho;
    S.pvalunit(cc) = pval;
end

if opts.plot
    figure(1); clf
    plot(rpc + (0:(opts.ndim-1)), 'k')
    hold on
    plot(runningspeed/max(runningspeed) - 1)
    xlabel('Trials')
end