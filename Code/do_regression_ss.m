function varargout = do_regression_ss(D, cid, fout, overwrite)
% do_regression_ss(D, cid, fout)
% varargout = do_regression_ss(D, cid, fout, overwrite)

if nargin < 4
    overwrite = false;
end

fprintf('fitting %d\n', cid)

fname = fullfile(fout, sprintf('%s_%d.mat', D.subj, cid));

if ~overwrite && exist(fname, 'file')
    if nargout > 0
        varargout = cell(nargout, 1);
    end

    return
end

rng(1) % for reproducibility

opts = struct();
opts.randomize_folds = true;
opts.folds = 5;
opts.ntents = 20;
opts.nlfun = @nlfuns.logexp1;

%% Bin spikes and behavior
[stim, robs, behavior, unitopts] = bin_ssunit(D, cid, 'plot', false);

direction = stim{1};
freq = stim{3};
runspeed = nanmean(behavior{1},2); %#ok<*NANMEAN>
pupil = nanmean(behavior{3},2);
if all(isnan(pupil))
    pupil = zeros(size(runspeed)); % ignore pupil: no pupil data
end

good_ix = ~(isnan(direction) | isnan(runspeed) | isnan(pupil));

R = mean(robs(:,unitopts.lags>0),2)/unitopts.binsize;
R = R(good_ix);

direction = direction(good_ix);
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


%% build design matrix
fullR = Xstim;
regLabels = [{'Stim'}];
regIdx = ones(1, size(Xstim,2));
k = 1;

% add additional covariates
label = 'Drift';
num_tents = opts.ntents;
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

% build cross validation indices
xidxs = regression.xvalidationIdx(nt, opts.folds, opts.randomize_folds);


[rho, pval] = corr(R, runspeed, 'Type', 'Spearman');

Rpred_indiv = struct();
Rpred_indiv.data.cid = cid;
Rpred_indiv.data.R = R;
Rpred_indiv.data.runspeed = runspeed;
Rpred_indiv.data.direction = direction;
Rpred_indiv.data.freq = freq;
Rpred_indiv.data.runrho = rho;
Rpred_indiv.data.runrhop = pval;


% restLabels = {{'RunSpeed'} {'Drift'} {'IsRun'}};
modelNames = {'Stim', ...
    'RunNoGain', ...
    'DriftNoGain', ...
    'RunningSpeedGain', ...
    'SlowDrift', ...
    'RunningGain', ...
    'RunningGainNoDrift'};

restLabels = {{''}, ...
    {'RunSpeed'}, ...
    {'Drift'}, ...
    {'RunSpeed', 'Drift'}, ...
    {'Drift'}, ...
    {'IsRun', 'Drift'}, ...
    {'IsRun'}};

GainLabels = {'', ...
    '', ...
    '', ...
    'RunSpeed', ...
    'Drift', ...
    'IsRun', ...
    'IsRun'};

for iModel = 1:numel(modelNames)
    restLabel = restLabels{iModel};
    GainLabel = GainLabels{iModel};
    StimLabel  = 'Stim';

    Rpred =  nan(size(R));
    Gdrive = nan(size(R));
    Rpred_indiv.(modelNames{iModel}).gainLabel = GainLabel;
    Rpred_indiv.(modelNames{iModel}).restLabels = restLabel;
    Rpred_indiv.(modelNames{iModel}).Beta = cell(opts.folds,1);
    Rpred_indiv.(modelNames{iModel}).Ridge = nan(opts.folds,1);
    Rpred_indiv.(modelNames{iModel}).Gains = nan(opts.folds,2);
    Rpred_indiv.(modelNames{iModel}).Offset = nan(opts.folds,1);

    %         StimLabel, GainLabel, restLabel
    for ifold = 1:opts.folds

        %             [Betas, Gain, Ridge, Rhat, ~, ~, gdrive] = AltLeastSqGainModelFmin(fullR, R, xidxs{ifold,1}, regIdx, regLabels, StimLabel, GainLabel, restLabel, [],[],opts.nlfun);
        [Betas, Gain, Ridge, Rhat, ~, ~, gdrive]         = AltLeastSqGainModel(fullR, R, xidxs{ifold,1}, regIdx, regLabels, StimLabel, GainLabel, restLabel);
        
        fullix = ismember(regIdx, find(ismember(regLabels, [StimLabel, GainLabel, restLabel])));
        fitcovariatesix = regIdx(fullix);
        fitcovariates = unique(fitcovariatesix);
        for icov = 1:numel(fitcovariates)
            w = Betas(1+find(fitcovariatesix==fitcovariates(icov)));
            what = fullR(:,regIdx==fitcovariates(icov))*w;
            
            if ifold==1
                Rpred_indiv.(modelNames{iModel}).(regLabels{fitcovariates(icov)}) = struct('weights', w(:), 'rpred', what);
            else
                Rpred_indiv.(modelNames{iModel}).(regLabels{fitcovariates(icov)}).weights = [Rpred_indiv.(modelNames{iModel}).(regLabels{fitcovariates(icov)}).weights w(:)];
                Rpred_indiv.(modelNames{iModel}).(regLabels{fitcovariates(icov)}).rpred(xidxs{ifold,2}) = what(xidxs{ifold,2});
            end
        end
        
        Rpred(xidxs{ifold,2}) = Rhat(xidxs{ifold,2});
        Gdrive(xidxs{ifold,2}) = gdrive(xidxs{ifold,2});


        Rpred_indiv.(modelNames{iModel}).Offset(ifold) = Betas(1);
        Rpred_indiv.(modelNames{iModel}).Beta{ifold} = Betas(2:end);
        Rpred_indiv.(modelNames{iModel}).Gains(ifold,:) = Gain;
        Rpred_indiv.(modelNames{iModel}).Ridge(ifold) = Ridge;
    end

    Rpred_indiv.(modelNames{iModel}).Rpred = Rpred;
    Rpred_indiv.(modelNames{iModel}).Gdrive = Gdrive;

    % evaluate model
    Rpred_indiv.(modelNames{iModel}).Rsquared = rsquared(R, Rpred_indiv.(modelNames{iModel}).Rpred(:)); %compute explained variance
    [Rpred_indiv.(modelNames{iModel}).CC, Rpred_indiv.(modelNames{iModel}).CCpval]  = corr(R, Rpred_indiv.(modelNames{iModel}).Rpred(:)); %compute explained variance

    [Rpred_indiv.(modelNames{iModel}).runrho, Rpred_indiv.(modelNames{iModel}).runrhop] = corr(Rpred, runspeed, 'Type', 'Spearman');

end
fprintf('Done\n')

save(fname, '-v7.3', 'Rpred_indiv', 'opts')

if nargout > 0
    varargout{1} = Rpred_indiv;

    if nargout > 1
        varargout{2} = opts;
    end
end



