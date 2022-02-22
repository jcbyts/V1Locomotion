
%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';

subjects = {'mouse', 'marmoset'};
nsubjs = numel(subjects);

%% Load analyses from fig_main.m
afname = 'output/MainAnalysisUnweighted.mat';
if ~exist(afname, 'file')
    error('fig_regression_analysis: you must run fig_main first output/MainAnalysisUnweighted.mat is in your path')
end
Stat = load(afname);

%% check significant units with regression
fid = 1;
fout = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'regression_ss');

isubj = 1;
subject = subjects{isubj};

D = load_subject(subject);
D.subj = subject;

sigunits = Stat.(subject).cid(Stat.(subject).runrhop < 0.05);

%%
parfor cc = 1:numel(sigunits)
    cid = sigunits(cc);
    do_regression_ss(D, cid, fout, true)
end


%%
fout = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'regression_ss');
S = struct();

for isubj = 1:2
    subject = subjects{isubj};

    flist = dir(fullfile(fout, [subject '*.mat']));
    cids = arrayfun(@(x) str2double(cell2mat(regexp(x.name, '\d+', 'match'))), flist);
    rpred = cell(max(cids),1);

    for i = 1:numel(flist)

        load(fullfile(flist(i).folder, flist(i).name));
        rpred{Rpred_indiv.data.cid} = Rpred_indiv;
    end

%     iix = ~cellfun(@isempty, rpred);

    
    models2compare = {'RunningSpeedGain', 'Stim', 'RunNoGain', 'DriftNoGain', 'SlowDrift', 'RunningGain', 'RunningGainNoDrift'};
    for imodel = 1:numel(models2compare)
        model = models2compare{imodel};


        S.(subject).(model).rsquared = cellfun(@(x) x.(model).Rsquared, rpred(cids));
        S.(subject).(model).runrho = cellfun(@(x) x.(model).runrho, rpred(cids));
        S.(subject).(model).runrhop = cellfun(@(x) x.(model).runrhop, rpred(cids));
        S.(subject).(model).cids = cellfun(@(x) x.data.cid, rpred(cids));
    end

    model = 'data';
    S.(subject).(model).runrho = cellfun(@(x) x.(model).runrho, rpred(cids));
    S.(subject).(model).runrhop = cellfun(@(x) x.(model).runrhop, rpred(cids));


    figure(isubj); clf
    cmap = getcolormap(subject, false);


    subplot(1,2,1)
    m1 = 'Stim';
    m2 = 'RunNoGain';
    r21 =  S.(subject).(m1).rsquared;
    r22 = S.(subject).(m2).rsquared; 
    iix = max(r21, r22) > 0;
    r21 = r21(iix);
    r22 = r22(iix);
    fprintf('%s: %02.2f %% of units better explained by %s than %s\n', subject, mean(r22 > r21)*100, m2, m1)

    plot(r21, r22, '.', 'Color', cmap(6,:))
    hold on
    l = refline(1, 0); l.Color = 'k';
    xlabel('$r^2$ (Stimulus Only)', 'Interpreter', 'Latex')
    ylabel('$r^2$ (Stimulus + Running)', 'Interpreter', 'latex')
    plot.offsetAxes(gca)

    
    subplot(1,2,2)
    m1 = 'RunNoGain';
    m2 = 'DriftNoGain';
    r21 =  S.(subject).(m1).rsquared;
    r22 = S.(subject).(m2).rsquared; 
    iix = max(r21, r22) > 0;
    r21 = r21(iix);
    r22 = r22(iix);
    fprintf('%s: %02.2f %% of units better explained by %s than %s\n', subject, mean(r22 > r21)*100, m2, m1)

    plot(r21, r22, '.', 'Color', cmap(6,:))

    hold on
    l = refline(1, 0); l.Color = 'k';
    xlabel('$r^2$ (Stimulus + Running)', 'Interpreter', 'Latex')
    ylabel('$r^2$ (Stimulus + Drift)', 'Interpreter', 'latex')
    
    xlim([-.1 1])
    ylim([-.1 1])
    plot.offsetAxes(gca)

    plot.formatFig(gcf, [4,2], 'nature')
    saveas(gcf, fullfile(figdir, sprintf('regression_%s.pdf', subject)))

    figure(10+isubj); clf
    cmap = getcolormap(subject, false);


    subplot(1,2,1)
    m1 = 'RunNoGain';
    m2 = 'RunningGainNoDrift';
    r21 =  S.(subject).(m1).rsquared;
    r22 = S.(subject).(m2).rsquared; 
    iix = max(r21, r22) > 0;
    r21 = r21(iix);
    r22 = r22(iix);
    fprintf('%s: %02.2f %% of units better explained by %s than %s\n', subject, mean(r22 > r21)*100, m2, m1)

    plot(r21, r22, '.', 'Color', cmap(6,:))
    hold on
    l = refline(1, 0); l.Color = 'k';
    xlabel('$r^2$ (Running (Additive))', 'Interpreter', 'Latex')
    ylabel('$r^2$ (Running (Gain))', 'Interpreter', 'latex')
    
    
    
    subplot(1,2,2)
    m1 = 'DriftNoGain';
    m2 = 'SlowDrift';
    r21 =  S.(subject).(m1).rsquared;
    r22 = S.(subject).(m2).rsquared; 
    iix = max(r21, r22) > 0;
    r21 = r21(iix);
    r22 = r22(iix);
    fprintf('%s: %02.2f %% of units better explained by %s than %s\n', subject, mean(r22 > r21)*100, m2, m1)

    plot(r21, r22, '.', 'Color', cmap(6,:))

    hold on
    l = refline(1, 0); l.Color = 'k';
    xlabel('$r^2$ (Drift (Additive))', 'Interpreter', 'Latex')
    ylabel('$r^2$ (Drift (Gain))', 'Interpreter', 'latex')
    
    xlim([-.1 1])
    ylim([-.1 1])
end
%%

% i = i + 1;
i = 1;
m1 = 'RunningGainNoDrift';
m2 = 'DriftNoGain';

goodcc = find(S.(subject).Stim.rsquared>.2);

r21 = S.(subject).(m1).rsquared(goodcc);
r22 = S.(subject).(m2).rsquared(goodcc);


figure(1); clf
plot(r21, r22, '.')

[~, ind] = sort(r22-r21, 'descend');

unitlist = goodcc(ind);

%%

figure(2); clf
subplot(2, 1, 1)
plot(rpred{cids(i)}.data.R, 'k'); hold on
plot(rpred{cids(i)}.(m1).Rpred, 'r')
plot(rpred{cids(i)}.(m1).Gdrive, 'g')
title(rpred{cids(i)}.(m1).Rsquared)

subplot(2,1,2)
plot(rpred{cids(i)}.data.R, 'k'); hold on
plot(rpred{cids(i)}.(m2).Rpred, 'r')
plot(rpred{cids(i)}.(m2).Gdrive, 'g')
title(rpred{cids(i)}.(m2).Rsquared)


%%

subject = 'marmoset';
D = load_subject(subject);
D.subj = subject;
cids = S.(subject).Stim.cids;
%% refit this unti
i = 1; %i + 1;

unitId = cids(unitlist(i));
[Rpred_indiv, opts] = do_regression_ss(D, unitId, fout, true);

rpred{cids(i)} = Rpred_indiv;

m1 = 'DriftNoGain';

%%
m2 = 'DriftNoGain';
m1 = 'RunNoGain';

figure(11); clf
cmap = getcolormap(subject, false);

NT = numel(Rpred_indiv.data.R);
bctrs = linspace(0, NT, opts.ntents);
Bt = tent_basis((1:NT)', bctrs);
xd = [1 min(1200,NT)];

directions = unique(Rpred_indiv.data.direction);
nd = numel(directions);

% stimulus condition
axes('Position', [.1 .82, .7 .1])
% plot(Rpred_indiv.data.direction, '.k')
plot.raster(1:NT, Rpred_indiv.data.direction, 22);
set(gca, 'YTick', 0:90:max(directions))
ylabel('Direction')
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)
ylim([0 max(directions)])

% stimulus weights
axes('Position', [.85, .82, .12, .1])
w = mean(Rpred_indiv.(m1).Stim.weights,2);
w = reshape(w, nd, []);
for i = 1:size(w,2)
    plot(directions, w(:,i), 'Color', cmap(i*2,:)); hold on
end
set(gca, 'XTick', 0:90:max(directions))
xlim([0 max(directions)])
set(gca, 'box', 'off')
labels = arrayfun(@(x) sprintf('%d cyc/deg', x), unique(Rpred_indiv.data.freq), 'uni', 0);
legend(labels{:})

% running speed
axes('Position', [.1 .75, .7 .05])
plot(Rpred_indiv.data.runspeed, 'Color', repmat(.5, 1, 3))
ylabel('Running Speed (cm/s)')
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)


axes('Position', [.1 .67, .7 .05])
clrs = gray(size(Bt,2)+10);
clrs(1:2,:) = [];
for i = 1:size(Bt,2)
    plot(Bt(:,i), 'Color', clrs(i,:)); hold on
end
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)

axes('Position', [.1 .57, .7 .05])
plot(Rpred_indiv.(m1).Stim.rpred, 'Color', 'k')
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)

axes('Position', [.1 .47, .7 .05])
plot(Rpred_indiv.(m1).RunSpeed.rpred, 'Color', 'k')
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)

axes('Position', [.1 .4, .7 .05])
clrs = gray(size(Bt,2));
% w = mean(Rpred_indiv.(m2).Drift.weights,2);
% for i = 1:size(Bt,2)
%     plot(bctrs(i)*[1 1], [0 w(i)], 'k')
%     plot(Bt(:,i)*w(i), 'Color', clrs(i,:)); hold on
% end
% plot(Bt*w, 'Color', 'k'); hold on
plot(Rpred_indiv.(m2).Drift.rpred, 'k')
set(gca, 'box', 'off', 'XTickLabel', '')
xlim(xd)


axes('Position', [.1 .28, .7 .1])
stairs(Rpred_indiv.data.R, 'k'); hold on
plot(Rpred_indiv.(m1).Rpred)
ylabel('Firing Rate')
xlim(xd)
title(Rpred_indiv.(m1).Rsquared)
set(gca, 'box', 'off', 'XTickLabel', '')

axes('Position', [.1 .12, .7 .1])
stairs(Rpred_indiv.data.R, 'k'); hold on
plot(Rpred_indiv.(m2).Rpred)
ylabel('Firing Rate')
xlim(xd)
title(Rpred_indiv.(m2).Rsquared)
set(gca, 'box', 'off', 'XTick', 0:100:200)


xlabel('Trial')
plot.formatFig(gcf, [4 8], 'nature')
saveas(gcf, fullfile(figdir, sprintf('regression_example_%s_%d.pdf', subject, unitId)))
% , plot(Rpred_indiv.(m1).Drift.weights)


%%
unitId = cids(i);

figure(10); clf
subplot(2, 1, 1)
plot(rpred{cids(i)}.data.R, 'k'); hold on
plot(rpred{cids(i)}.(m1).Rpred, 'r')
plot(rpred{cids(i)}.(m1).Gdrive, 'g')
title(rpred{cids(i)}.(m1).Rsquared)

subplot(2,1,2)
plot(rpred{cids(i)}.data.R, 'k'); hold on
plot(rpred{cids(i)}.(m2).Rpred, 'r')
plot(rpred{cids(i)}.(m2).Gdrive, 'g')
title(rpred{cids(i)}.(m2).Rsquared)



%% fit example unit and show how the model works

%   stim:      {StimDir, StimSpeed, StimFreq};
%   robs:      Binned spikes;
%   behavior:  {runSpeed, GratPhase, PupilArea, Fixations};

[stim, robs, behavior, unitopts] = bin_ssunit(D, unitId, 'plot', false);

direction = stim{1};
speed = stim{2};
freq = stim{3};
runspeed = nanmean(behavior{1},2); %#ok<*NANMEAN>
pupil = nanmean(behavior{3},2);

good_ix = ~(isnan(direction) | isnan(runspeed) | isnan(pupil));

R = mean(robs(:,unitopts.lags>0),2)/unitopts.binsize;
R = R(good_ix);

direction = direction(good_ix);
speed = speed(good_ix);
freq = freq(good_ix);
runspeed = runspeed(good_ix);
pupil = pupil(good_ix);

nt = numel(runspeed);

directions = unique(direction(:))';
freqs = unique(freq(:))';

Dmat = directions==direction;
Fmat = freq==freqs;

nf = numel(freqs);
nd = numel(directions);

Xstim = nan(nt, nd*nf);

for f = 1:nf
    Xstim(:,(f-1)*nd + (1:nd)) = Dmat.*Fmat(:,f);
end


% build design matrix
fullR = Xstim;
regLabels = [{'Stim'}];
regIdx = ones(1, size(Xstim,2));
k = 1;

% add additional covariates
label = 'Drift';
num_tents = 20;
regLabels = [regLabels {label}];
k = k + 1;
Btents = tent_basis( (1:nt)', linspace(0, nt, num_tents));
regIdx = [regIdx repmat(k, [1, size(Btents,2)])];
fullR = [fullR Btents];

% add additional covariates
label = 'RunSpeed';
regLabels = [regLabels {label}];
k = k + 1;
regIdx = [regIdx k];
fullR = [fullR runspeed];

% add additional covariates
label = 'IsRun';
regLabels = [regLabels {label}];
k = k + 1;
regIdx = [regIdx k];
fullR = [fullR runspeed>3];

label = 'PupilArea';
regLabels = [regLabels {label}];
k = k + 1;
regIdx = [regIdx k];
fullR = [fullR pupil];

opts = struct();
opts.randomize_folds = true;
opts.folds = 5;

xidxs = regression.xvalidationIdx(nt, opts.folds, opts.randomize_folds);

% restLabels = {{'RunSpeed'} {'Drift'} {'IsRun'}};
restLabels = {{'RunSpeed', 'Drift'}, {'Drift'}, {'IsRun'}};
GainLabels = {'RunSpeed', 'Drift', 'IsRun'};
StimLabels = {'Stim', 'Stim', 'Stim'};

[rho, pval] = corr(R, runspeed, 'Type', 'Spearman');

Rpred =  nan(size(R,1),3);
Gdrive = nan(size(R,1),3);

for iModel = 1:3
    restLabel = restLabels{iModel};
    GainLabel = GainLabels{iModel};
    StimLabel  = StimLabels{iModel};


    for ifold = 1:opts.folds
        
        [Betas, Gain, Ridge, Rhat, Lgain, Lfull, gdrive] = AltLeastSqGainModelFmin(fullR, R, xidxs{ifold,1}, regIdx, regLabels, StimLabel, GainLabel, restLabel, [],[],@nlfuns.logexp1);
%         [Betas, Gain, Ridge, Rhat, Lgain, Lfull, gdrive] = AltLeastSqGainModel(fullR, R, xidxs{ifold,1}, regIdx, regLabels, StimLabel, GainLabel, restLabel);

        Rpred(xidxs{ifold,2}, iModel) = Rhat(xidxs{ifold,2});
        Gdrive(xidxs{ifold,2}, iModel) = gdrive(xidxs{ifold,2});
    end


    figure(iModel); clf
    subplot(3,1,1)
    plot(R, 'k'); hold on
    plot(runspeed, 'r')
    xlabel('Trial')
    title(['$\rho$ = ' num2str(rho,3) ', p= ' num2str(pval, 3)], 'Interpreter', 'latex')
    
    subplot(3,1,2)
    plot(R, 'k')
    hold on
    plot(Rpred(:, iModel), 'r')
    title(rsquared(R, Rpred(:,iModel)))
    subplot(3,1,3)
    plot(Gdrive(:, iModel), 'g'); hold on
    plot(runspeed/max(runspeed)*max(Gdrive(:, iModel)), 'k');
    
    [rhomod, pmod] = corr(Rpred(:, iModel), runspeed, 'Type', 'Spearman');
    title(['$\rho$ = ' num2str(rhomod,3) ', p= ' num2str(pmod, 3)], 'Interpreter', 'latex')

end

%%


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






%%
xidxs = xvalidationIdx(nt, opts.folds, opts.randomize_folds);
yhat = nan(nt,1);

what = nan(size(r,2)+1,opts.folds);
for i = 1:opts.folds
    xtrain = r(xidxs{i,1},:);
    ytrain = runningspeed(xidxs{i,1});
    xtest = r(xidxs{i,2},:);
    [what(:,i), ~, outfun] = outputNonlinLs(xtrain,ytrain, struct('display', 'off'));
    yhat(xidxs{i,2}) = outfun(xtest*what(2:end,i) + what(1,i));
end
