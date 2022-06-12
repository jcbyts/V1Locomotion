%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';

%% Basic summary of session running
thresh = 3; % running threshold

subjects = {'mouse', 'marmoset'};

nsubjs = numel(subjects);

runfrac = cell(nsubjs,1);

RunCorr = struct();

for isubj = 1:nsubjs
    subject = subjects{isubj};

    D = load_subject(subject, fpath);
    D.subject = subject;
    cmap = getcolormap(subject,false);

    sessnums = unique(D.sessNumTread);
    nsess = numel(sessnums);
    
    fprintf(1,"%d sessions\n", nsess);
    
    RunCorr.(subject) = cell(nsess,1);
    opts = struct();
    opts.save = true;
    opts.prewin = -0.05;
    opts.postwin = 0.05;
    opts.spike_rate_thresh = 1;

    for isess = 1:nsess
        fprintf(1,'%d/%d session\n', isess, nsess);
        RunCorr.(subject){isess} = running_vs_spikePC(D, sessnums(isess), opts);
    end
end