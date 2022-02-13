%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';

%% Basic summary of session running
thresh = 3; % running threshold
subjects = {'gru', 'brie', 'allen'};

nsubjs = numel(subjects);

runfrac = cell(nsubjs,1);

figure(1); clf
figure(2); clf

RunCorr = struct();

for isubj = 1:nsubjs
    subject = subjects{isubj};

    D = load_subject(subject, fpath);
    cmap = getcolormap(subject,false);

    sessnums = unique(D.sessNumTread);
    nsess = numel(sessnums);
    
    fprintf("%d sessions\n", nsess)
    
    RunCorr.(subject) = cell(nsess,1);
    
    for isess = 1:nsess
        RunCorr.(subject){isess} = running_vs_spikePC(D, sessnums(isess));
    end
end

%%
figure(1); clf
trial_thresh = 300; % only include sessions with more than this number

for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    npcs = 1;
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));

    for ipc = 1:npcs
        rho = cellfun(@(x) x.rho(ipc), RunCorr.(subject)(nt > trial_thresh));
        pval = cellfun(@(x) x.pval(ipc), RunCorr.(subject)(nt > trial_thresh));
        
        subplot(nsubjs, npcs, (isubj-1)*npcs + ipc)
        histogram(rho, 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
        histogram(rho(pval < 0.05), 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
        plot(median(rho), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:))
        xlim([-1 1])
        
        [p, ~, pstat] = signrank(rho);
        fprintf('%s, median=%02.3f, pval=%0.5f (rank=%02.5f)\n', subject, median(rho),p,pstat.signedrank);

        plot.offsetAxes(gca)
        ylabel('# sessions')
        text(.2, .7*max(ylim), sprintf('%02.3f (p = %02.5f, %d, %d)', median(rho), p, pstat.signedrank, numel(rho)), 'FontSize',6, 'FontName', 'Helvetica')
    end
end

xlabel('Corr. w/ Running')
plot.formatFig(gcf, [1.37 4.56], 'nature')
saveas(gcf,fullfile(figdir, 'runpc_corr_hist.pdf'))

%% plot individual sessions

sort_by_running = false;

for isubj = 1:nsubjs
    subject = subjects{isubj};
    ntrials = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));

    cmap = getcolormap(subject, false);

    goodsess = find(ntrials > trial_thresh);

    rho = cellfun(@(x) x.rho(ipc), RunCorr.(subject)(goodsess));

    [~, ind] = sort(rho);
    ind = goodsess(ind);

    sessnum = [ind(1) ind(ceil(numel(ind)/2)) ind(numel(ind))];
    for i = 1:3
        isess = sessnum(i);

        robs = RunCorr.(subject){isess}.robs;
        runspeed = RunCorr.(subject){isess}.runningspeed;

        rpc = RunCorr.(subject){isess}.rpc;
        nt = size(robs,1);

        [~, uind] = sort(RunCorr.(subject){isess}.rhounit);

        robs = robs(:,uind);
        if sort_by_running
            [runspeed,tind] = sort(runspeed); %#ok<*UNRCH> 
            robs = robs(tind,:);
            rpc = rpc(tind,:);
        end

        figure(isubj*10 + i); clf
        axes('Position', [.1 .4 .8 .5])
        imagesc(robs'); hold on
        title(sprintf('rho: %02.3f', RunCorr.(subject){isess}.rho(1)))
        nt = max(nt, trial_thresh);
        nt = nt + 5;

        xlim([1 nt])
        axis off
        hold on
        plot([1 1], [0 10], 'k', 'Linewidth', 1)
        colormap(cmap)

        axes('Position', [.1 .25 .8 .15])
        plot(rpc(:,1), 'Color', cmap(6,:));
        xlim([1 nt])
        axis off
    
        axes('Position', [.1 .08 .8 .15])
        plot(runspeed, 'Color', repmat(.5, 1, 3))
        hold on
        xlim([1 nt])
        plot([1 51], -1*[1 1], 'k', 'Linewidth', 1)
        text(5, -7, sprintf('%d trials', 50), 'FontSize',6, 'FontName', 'Helvetica')
        mxsp = nanmax(runspeed)/2;
        mxsp = roundn(mxsp, 1);

        plot([1 1]*(nt-1), [0 mxsp], 'k', 'Linewidth', 1)
        text(nt, 0, sprintf('%d cm/s', mxsp), 'FontSize',6, 'FontName', 'Helvetica', 'Rotation', 90);
        axis off
        
        plot.formatFig(gcf, [3.93 1.2], 'nature')
        saveas(gcf,fullfile(figdir, sprintf('runpc_%s_%d_%d.pdf', subject, i, isess)))
    end
end



