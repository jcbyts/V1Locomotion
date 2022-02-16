function do_regression_ss(D, cid, fout)

fname = fullfile(fout, sprintf('%s_%d.mat', D.subj, cid));

if exist(fname, 'file')
    return
end


try
    
    [Stim, Robs] = bin_spikes_as_trials(D, cid, 'plot', true, 'binsize', 1/60, 'win', [-.2, .2]);

    if (mean(Robs(:)) / Stim.bin_size) < 1
        return
    end

    opts = struct();
    folds = 5;
    opts.use_spikes_for_dfs = false;
    opts.folds = folds;
    opts.trialidx = (1:numel(Stim.grating_onsets))';
    opts.truetrialidx = Stim.trial_list;
    opts.spike_smooth = 5;

    Robs = filtfilt(ones(opts.spike_smooth,1)/opts.spike_smooth, 1, Robs);
    Robs = Robs(:);
    NC = 1;
    
    %% build the design matrix components
    % build design matrix
    assert(Stim.bin_size==1/60, 'Parameters have only been set for 60Hz sampling')
    opts.collapse_speed = true;
    opts.collapse_phase = true;
    opts.include_onset = false;
    opts.stim_dur = median(D.GratingOffsets(opts.trialidx)-D.GratingOnsets(opts.trialidx)) + 0.2; % length of stimulus kernel
    opts.use_sf_tents = false;

    stim_dur = ceil((opts.stim_dur)/Stim.bin_size);
    opts.stim_ctrs = [2:5:10 15:15:stim_dur];
    if ~isfield(opts, 'stim_ctrs')
        opts.stim_ctrs = [0:5:10 15:10:stim_dur-2];
    end
    Bt = tent_basis(0:stim_dur+15, opts.stim_ctrs);

    [X, opts] = build_design_matrix(Stim, opts);

    % concatenate full design matrix
    label = 'Stim';
    regLabels = {label};
    k = 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = repmat(k, 1, size(X_,2)); %#ok<REPMAT>
    fullR = X_;

    if opts.include_onset
        label = 'Stim Onset';
        regLabels = [regLabels {label}];
        k = k + 1;
        X_ = X{ismember(opts.Labels, label)};
        regIdx = [regIdx repmat(k, [1, size(X_,2)])];
        fullR = [fullR X_];
    end

    % add additional covariates
    label = 'Drift';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    label = 'Saccade';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    label = 'Running';
    regLabels = [regLabels {label}];
    k = k + 1;
    X_ = X{ismember(opts.Labels, label)};
    regIdx = [regIdx repmat(k, [1, size(X_,2)])];
    fullR = [fullR X_];

    isrunning = reshape(Stim.tread_speed(:, opts.trialidx), [], 1) > opts.run_thresh;
    isrunning = sign(isrunning - .5);

    zpupil = reshape(Stim.eye_pupil(:, opts.trialidx), [], 1) / nanstd(reshape(Stim.eye_pupil(:, opts.trialidx), [], 1)); %#ok<*NANSTD>
    zpupil = zpupil - nanmean(zpupil); %#ok<*NANMEAN>
    ispupil = zpupil > opts.pupil_thresh;
    ispupil = sign(ispupil - .5);

    regLabels = [regLabels {'Is Running'}];
    k = k + 1;
    regIdx = [regIdx k];
    fullR = [fullR isrunning(:)];

    regLabels = [regLabels {'Is Pupil'}];
    k = k + 1;
    regIdx = [regIdx k];
    fullR = [fullR ispupil(:)];

    assert(size(fullR,2) == numel(regIdx), 'Number of covariates does not match Idx')
    assert(numel(unique(regIdx)) == numel(regLabels), 'Number of labels does not match')

    for i = 1:numel(regLabels)
        label= regLabels{i};
        cov_idx = regIdx == find(strcmp(regLabels, label));
        fprintf('[%s] has %d parameters\n', label, sum(cov_idx))
    end

    %% Find valid time range using stim gain model

    Rpred_indiv = struct();
    Rpred_indiv.data = struct();

    % try different models
%     modelNames = {'nostim', 'stim', 'stimsac', 'stimrun', 'stimrunsac', 'drift'};
%     excludeParams = { {'Stim', 'Stim Onset'}, {'Running', 'Saccade'}, {'Running'}, {'Saccade'}, {}, {'Stim','Stim Onset','Running','Saccade'} };
%     alwaysExclude = {'Stim R', 'Stim S', 'Is Running', 'Is Pupil'};


    modelNames = {'nostim', 'stimsac', 'stimrunsac'};
    excludeParams = { {'Stim', 'Stim Onset'}, {'Running'}, {'Saccade'}, {}};
    alwaysExclude = {'Stim R', 'Stim S', 'Is Running', 'Is Pupil'};

    % models2fit = {'drift', 'stim'};
    models2fit = modelNames;

    Ntotal = size(Robs,1);
    Rpred_indiv.data.indices = false(Ntotal, NC);
    Rpred_indiv.data.Robs = Robs;

    for iModel = find(ismember(modelNames, models2fit))
        Rpred_indiv.(modelNames{iModel}).Rpred = nan(Ntotal,NC)';
        Rpred_indiv.(modelNames{iModel}).Offset = nan(folds,NC);
        Rpred_indiv.(modelNames{iModel}).Beta = cell(folds,1);
        Rpred_indiv.(modelNames{iModel}).Ridge = nan(1, NC);
        Rpred_indiv.(modelNames{iModel}).Rsquared = nan(1,NC);
        Rpred_indiv.(modelNames{iModel}).CC = nan(1,NC);
    end

    % Fit Gain models for running and pupil
    GrestLabels = [{'Stim Onset'}    {'Drift'}    {'Saccade'}];
    
    GainModelNames = {'RunningGain', ...
        'PupilGain', ...
        'DriftGain'};
    GainTerm = {'Is Running',...
        'Is Pupil',...
        'Drift'};

    for iModel = 1:numel(GainTerm)

        labelIdx = ismember(regLabels, [{'Stim'} GrestLabels]);
        covIdx = regIdx(ismember(regIdx, find(labelIdx)));
        covLabels = regLabels(labelIdx);
        [~, ~, covIdx] = unique(covIdx);

        Rpred_indiv.(GainModelNames{iModel}).covIdx = covIdx;
        Rpred_indiv.(GainModelNames{iModel}).Beta = cell(folds,1);
        Rpred_indiv.(GainModelNames{iModel}).Offset = nan(folds, NC);
        Rpred_indiv.(GainModelNames{iModel}).Gains = nan(folds, 2, NC);
        Rpred_indiv.(GainModelNames{iModel}).Labels = covLabels;
        Rpred_indiv.(GainModelNames{iModel}).Rpred = nan(NC, Ntotal);
        Rpred_indiv.(GainModelNames{iModel}).Gdrive = nan(NC, Ntotal);
        Rpred_indiv.(GainModelNames{iModel}).Ridge = nan(folds, NC);
        Rpred_indiv.(GainModelNames{iModel}).Rsquared = nan(1,NC);
        Rpred_indiv.(GainModelNames{iModel}).CC = nan(1,NC);

        ndim = numel(covIdx);
        for ifold = 1:folds
            Rpred_indiv.(GainModelNames{iModel}).Beta{ifold} = zeros(ndim, NC);
        end
    end

   

    tread_speed = reshape(Stim.tread_speed(:,opts.trialidx), [], 1);
    good_inds = find(~isnan(tread_speed));
    tread_speed = tread_speed(good_inds);

    % Build cv indices that are trial-based
    Rpred_indiv.data.indices(good_inds) = true;

    good_trials = (1:numel(Stim.grating_onsets))';  %find(dfsTrials(:,cc));
    num_trials = numel(good_trials);

    folds = 5;
    n = numel(Stim.trial_time);
    T = size(Robs,1);

    dataIdxs = true(folds, T);
    rng(1)

    trorder = randperm(num_trials);

    for t = 1:(num_trials)
        i = mod(t, folds);
        if i == 0, i = folds; end
        dataIdxs(i,(good_trials(trorder(t))-1)*n + (1:n)) = false;
    end

    % good_inds = intersect(good_inds, find(sum(dataIdxs)==(folds-1)));
    fprintf('%d good samples\n', numel(good_inds))

    R = Robs(good_inds);
    X = fullR(good_inds,:);

    for iModel = find(ismember(modelNames, models2fit))

        fprintf('Fitting Model [%s]\n', modelNames{iModel})

        exclude = [excludeParams{iModel} alwaysExclude];
        modelLabels = setdiff(regLabels, exclude); %#ok<*NASGU>
        %     evalc("[Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(fullR(good_inds,:), Robs(good_inds,:)', modelLabels, regIdx, regLabels, folds, dataIdxs);");
        % dataIdxs(:,df)

        [Vfull, fullBeta, ~, fullIdx, fullRidge, fullLabels] = crossValModel(X, R', modelLabels, regIdx, regLabels, folds, dataIdxs(:,good_inds));

        Rpred_indiv.(modelNames{iModel}).covIdx = fullIdx;
        Rpred_indiv.(modelNames{iModel}).Labels = fullLabels;
        Rpred_indiv.(modelNames{iModel}).Rpred(good_inds) = Vfull;
        Rpred_indiv.(modelNames{iModel}).Offset = cellfun(@(x) x(1), fullBeta(:));
        for i = 1:numel(fullBeta)
            fullBeta{i}(1) = [];
            Rpred_indiv.(modelNames{iModel}).Beta{i} = fullBeta{i}; %#ok<*NODEF>
        end

        Rpred_indiv.(modelNames{iModel}).Ridge = fullRidge;
        Rpred_indiv.(modelNames{iModel}).Rsquared = rsquared(R, Vfull'); %compute explained variance
        [Rpred_indiv.(modelNames{iModel}).CC, Rpred_indiv.(modelNames{iModel}).CCpval] = corr(R, Vfull'); %modelCorr(R,Vfull,1); %compute explained variance
    end

    Lgain = nan;
    Lfull = nan;

    for iModel = 1:numel(GainTerm)
        for ifold = 1:folds
            train_inds = find(dataIdxs(ifold,good_inds))';
            test_inds = find(~dataIdxs(ifold,good_inds))';

            %             evalc("[Betas, Gain, Ridge, Rhat, Lgain, Lfull] = AltLeastSqGainModel(fullR(good_inds,:), Robs(good_inds,cc), train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), restLabels, Lgain, Lfull);");
            [Betas, Gain, Ridge, Rhat, Lgain, Lfull, gdrive] = AltLeastSqGainModel(X, R, train_inds, regIdx, regLabels, {'Stim'}, GainTerm(iModel), GrestLabels, Lgain, Lfull);

            Rpred_indiv.(GainModelNames{iModel}).Offset(ifold) = Betas(1);
            Rpred_indiv.(GainModelNames{iModel}).Beta{ifold} = Betas(2:end);
            Rpred_indiv.(GainModelNames{iModel}).Gains(ifold,:) = Gain;
            Rpred_indiv.(GainModelNames{iModel}).Ridge(ifold) = Ridge;
            Rpred_indiv.(GainModelNames{iModel}).Rpred(good_inds(test_inds)) = Rhat(test_inds);
            Rpred_indiv.(GainModelNames{iModel}).Gdrive(good_inds(test_inds)) = gdrive(test_inds);
        end
        % evaluate model
        Rpred_indiv.(GainModelNames{iModel}).Rsquared = rsquared(R, Rpred_indiv.(GainModelNames{iModel}).Rpred(good_inds)'); %compute explained variance
        [Rpred_indiv.(GainModelNames{iModel}).CC,Rpred_indiv.(GainModelNames{iModel}).CCpval]  = corr(R, Rpred_indiv.(GainModelNames{iModel}).Rpred(good_inds)'); %compute explained variance
        fprintf('Done\n')
    end

        %     drawnow


%     save(fname, '-v7.3', 'Rpred_indiv', 'opts')

end

%%
iix = Rpred_indiv.data.indices;
figure(1); clf

models2compare = {'stimsac', 'stimrunsac', 'RunningGain', 'DriftGain', 'PupilGain'};
nmodels = numel(models2compare);
for imodel = 1:nmodels
    subplot(nmodels, 1, imodel)
modelname = models2compare{imodel};
plot(Rpred_indiv.data.Robs(iix), 'k'); hold on
plot(Rpred_indiv.(modelname).Rpred(iix), 'r')
if isfield(Rpred_indiv.(modelname), 'Gdrive')
    plot(Rpred_indiv.(modelname).Gdrive(iix), 'g')
end
title(sprintf('%s: r2:%02.3f, rho:%02.3f', modelname, Rpred_indiv.(modelname).Rsquared, Rpred_indiv.(modelname).CC))
end

