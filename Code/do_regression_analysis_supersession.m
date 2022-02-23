function do_regression_analysis_supersession(opts)
% [Stim, opts, Rpred, Running] = do_regression_analysis(D)

if nargin < 1
    opts = struct();
end

defopts = struct();
defopts.use_parfor = false;
opts = mergeStruct(defopts, opts);


%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');

subjects = {'mouse', 'marmoset'};
nsubjs = numel(subjects);

%% Load analyses from fig_main.m
afname = 'output/MainAnalysisUnweighted.mat';
if ~exist(afname, 'file')
    error('fig_regression_analysis: you must run fig_main first output/MainAnalysisUnweighted.mat is in your path')
end
Stat = load(afname);

%% check significant units with regression
fout = fullfile(getpref('FREEVIEWING', 'HUKLAB_DATASHARE'), 'regression_ss');

for isubj = 1:nsubjs
    subject = subjects{isubj};

    D = load_subject(subject, fpath);
    D.subj = subject;

    sigunits = Stat.(subject).cid(Stat.(subject).runrhop < 0.05);

    %% refit regression
    if opts.use_parfor
        parfor cc = 1:numel(sigunits)
            cid = sigunits(cc);
            do_regression_ss(D, cid, fout, true)
        end
    else
        for cc = 1:numel(sigunits)
            cid = sigunits(cc);
            do_regression_ss(D, cid, fout, true)
        end
    end
end

