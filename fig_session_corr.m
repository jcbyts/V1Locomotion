%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';
fid = fopen('fig_session_corr.txt', 'w');

%% Basic summary of session running
thresh = 3; % running threshold

subjects = {'mouse', 'marmoset'};

nsubjs = numel(subjects);

runfrac = cell(nsubjs,1);

RunCorr = struct();

for isubj = 1:nsubjs
    subject = subjects{isubj};

    D = load_subject(subject, fpath);
    cmap = getcolormap(subject,false);

    sessnums = unique(D.sessNumTread);
    nsess = numel(sessnums);
    
    fprintf(fid,"%d sessions\n", nsess)
    
    RunCorr.(subject) = cell(nsess,1);
    
    for isess = 1:nsess
        fprintf(fid,'%d/%d session\n', isess, nsess)
        RunCorr.(subject){isess} = running_vs_spikePC(D, sessnums(isess));
    end
end

%% histogram of spike PC correlation with running
figure(1); clf
trial_thresh = 300; % only include sessions with more than this number
frac_run_thresh = [.1 .9];
nboot = 500;

rhos = cell(nsubjs,1);

for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    npcs = 1;
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2);

    for ipc = 1:npcs
        rho = cellfun(@(x) x.rho(ipc), RunCorr.(subject)(sessix));
        pval = cellfun(@(x) x.pval(ipc), RunCorr.(subject)(sessix));
        rhos{isubj} = rho;

        subplot(nsubjs, npcs, (isubj-1)*npcs + ipc)

        histogram(rho, 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
        histogram(rho(pval < 0.05), 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
        plot(median(rho), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:))
        xlim([-1 1])
        
        [p, ~, pstat] = signrank(rho);
        fprintf(fid, '%s, median=%02.3f, pval=%0.5f (rank=%02.5f)\n', subject, median(rho),p,pstat.signedrank);

        plot.offsetAxes(gca)
        ylabel('# sessions')
        text(.2, .7*max(ylim), sprintf('%02.3f (p = %02.5f, %d, %d)', median(rho), p, pstat.signedrank, numel(rho)), 'FontSize',6, 'FontName', 'Helvetica')
    end
end

[pval, ~, stats] = ranksum(rhos{1}, rhos{2});
fprintf(fid,'Testing whether %s and %s PC correlations come from distributions with different medians\n', subjects{:});
if pval < 0.05
    fprintf(fid,'%s and %s significantly differ\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

else
    fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
end

xlabel('Corr. w/ Running')
if nsubjs==3
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'runpc_corr_hist.pdf'))
else
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'runpc_corr_hist_marm.pdf'))
end


%% histogram of fraction running
figure(1); clf

% subjects = {'allen', 'gru', 'brie'};
fracsrun = cell(nsubjs,1);

for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2);

           
    subplot(nsubjs, 1, isubj)
    histogram(fracrun, 'binEdges', linspace(0, 1, 20), 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
    histogram(fracrun(sessix), 'binEdges', linspace(0, 1, 20), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
    plot(median(fracrun(sessix)), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:))
    xlim([0 1])
    fracsrun{isubj} = fracrun;

    ci = bootci(nboot, @median, fracrun(sessix));
    fprintf(fid,'%s, median=%02.3f, [%02.3f, %02.3f]\n', subject, median(rho),ci(1), ci(2));

    plot.offsetAxes(gca)
    ylabel('# sessions')
    text(.2, .7*max(ylim), sprintf('%02.3f [%02.3f, %02.3f]', median(rho), ci(1), ci(2)), 'FontSize',6, 'FontName', 'Helvetica')
end

xlabel('Fraction Running')
if nsubjs==3
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'frac_running_hist.pdf'))
else
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'frac_running_hist_marm.pdf'))
end

[pval, ~, stats] = ranksum(fracsrun{:});
fprintf(fid,'Testing whether %s and %s Fraction Running come from distributions with different medians\n', subjects{:});
if pval < 0.05
    fprintf(fid,'%s and %s significantly differ\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );

else
    fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
end




%% histogram of running decoding
figure(1); clf
r2_all = [];
fracrun_all = [];
subj_id = [];

% get bins for plotting
plot_bins = linspace(-.5, 1, 20);
for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2);

    r2 = cellfun(@(x) x.decoding_r2, RunCorr.(subject)(sessix));
            
    subplot(nsubjs, npcs, (isubj-1)*npcs + 1)
    histogram(r2, 'binEdges', plot_bins, 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
    plot(median(r2), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:))
    xlim(plot_bins([1 end]))
    
    [p, ~, pstat] = signrank(r2);
    fprintf(fid,'%s, median=%02.3f, pval=%0.5f (rank=%02.5f)\n', subject, median(r2),p,pstat.signedrank);
    
    plot.offsetAxes(gca)
    ylabel('# sessions')
    text(.2, .7*max(ylim), sprintf('%02.3f (p = %02.5f, %d, %d)', median(r2), p, pstat.signedrank, numel(r2)), 'FontSize',6, 'FontName', 'Helvetica')
    if p < 0.05
        text(median(r2), .95*max(ylim), '*', 'FontSize', 12)
    end

    r2_all = [r2_all; r2(:)];
    fracrun_all = [fracrun_all; fracrun(sessix)];
    subj_id = [subj_id; ones(sum(sessix),1)*isubj];

end

xlabel('Decoding Performance (r^2)')
if nsubjs==3
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'rundecoding_hist.pdf'))
else
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, 'rundecoding_hist_marm.pdf'))
end


figure(10); clf
h = [];
ax = gca;
ax.Position(3:4) = ax.Position(3:4)*.8;
for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    ix = subj_id == isubj;
    h(isubj) = plot(fracrun_all(ix), r2_all(ix), 'o', 'Color', 'w', 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 3); hold on
end
% legend(h, subjects, 'Location', 'BestOutside')
ylabel('Decoding Performance (r^2)')
xlabel('Fraction Running')
xd = xlim(ax);
yd = ylim(ax);
axtop = axes('Position', [ax.Position(1) ax.Position(2) + ax.Position(4)+.02, ax.Position(3), .15]);
for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    ix = subj_id == isubj;
    histogram(fracrun_all(ix), 'binEdges', linspace(xd(1), xd(end), 25), 'FaceColor', cmap(6,:), 'EdgeColor', cmap(6,:), 'FaceAlpha', .5); hold on
end
xlim(xd)

axright = axes('Position', [ax.Position(1) + ax.Position(3)+.02 ax.Position(2) .15 ax.Position(4)]);
for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    ix = subj_id == isubj;
    histogram(r2_all(ix), 'binEdges', linspace(yd(1), yd(end), 25), 'FaceColor', cmap(6,:), 'EdgeColor', cmap(6,:), 'FaceAlpha', .5); hold on
end

xlim(yd)
view(90,-90)

plot.formatFig(gcf, [1.37 1.37], 'nature')
set(ax, 'Box', 'off')
set(axtop, 'XTickLabel', [], 'XTick', get(ax, 'XTick'))
set(axright, 'XTickLabel', [], 'XTick', get(ax, 'YTick'))

if nsubjs==3
    saveas(gcf,fullfile(figdir, 'runfrac_vs_decoding.pdf'))
else
    saveas(gcf,fullfile(figdir, 'runfrac_vs_decoding_marm.pdf'))
end

[pval, ~, stats] = ranksum(r2_all(subj_id==1), r2_all(subj_id==2));
fprintf(fid,'Testing whether %s and %s Decoding Performance come from distributions with different medians\n', subjects{:});
if pval < 0.05
    fprintf(fid,'%s and %s significantly differ\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );

else
    fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );
end


%% plot individual sessions

sort_by_running = false;
sort_by_decoding = true;

for isubj = 1:nsubjs
    subject = subjects{isubj};
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2);

    cmap = getcolormap(subject, false);

    goodsess = find(sessix);
    
    if sort_by_decoding
        rho = cellfun(@(x) x.decoding_r2, RunCorr.(subject)(goodsess));
    else
        rho = cellfun(@(x) x.rho(1), RunCorr.(subject)(goodsess));
    end

    [~, ind] = sort(rho);
    ind = goodsess(ind);

    sessnum = [ind(1) ind(ceil(numel(ind)/2)) ind(numel(ind))];
    sessnum = ind;
    for i = 1:numel(sessnum)
        isess = sessnum(i);

        robs = RunCorr.(subject){isess}.robs;
        runspeed = RunCorr.(subject){isess}.runningspeed;
        runhat = RunCorr.(subject){isess}.decoding_runspeed;

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
        title(['$\rho:$ ' num2str(RunCorr.(subject){isess}.rho(1), 3), ',   $r^2:$' num2str(RunCorr.(subject){isess}.decoding_r2, 3)], 'interpreter', 'latex')
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
        plot(runspeed, 'Color', repmat(.1, 1, 3)); hold on
        plot(runhat, 'Color', cmap(6,:))

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


        figure(isubj*100 + i); clf
        
        iix = runspeed >= 0 & runhat >=0;
        plot(runspeed(iix), runhat(iix), 'ow', 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2); hold on
        xd = [min(runspeed(iix)) max(runspeed(iix)) + 10 - mod(max(runspeed(iix)),10)];
        plot(xd, xd, 'k')
        xlim(xd)
        ylim(xd)
        axis square
        plot.offsetAxes(gca)
        
        xlabel('True (cm s^{-1})')
        ylabel('Decoded (cm s^{-1})')
        plot.formatFig(gcf, [1.2 1.2], 'nature')
        saveas(gcf,fullfile(figdir, sprintf('rundecoding_%s_%d_%d.pdf', subject, i, isess)))


        iix = runspeed >= 0 & runhat >=0;
        clf
        h = qqplot(runspeed(iix), runhat(iix));
        h(1).MarkerEdgeColor = cmap(2,:);
        h(1).Marker = 'o';
        h(1).MarkerSize = 2;
        h(1).MarkerFaceColor = cmap(6,:);
        h(2).Color = 'k';
        h(3).Color = 'k';
        plot.formatFig(gcf, [1.2 1.2], 'nature')
        saveas(gcf,fullfile(figdir, sprintf('rundecodingqqplot_%s_%d_%d.pdf', subject, i, isess)))

    end
end

close all

fclose(fid);
