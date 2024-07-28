%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';

if ~exist('output', 'dir')
    mkdir('output')
end

save_output_for_modeling = true;
%% Basic summary of session running
trial_thresh = 250; % only include sessions with more than this number
frac_run_thresh = [.1 .9];
nboot = 500;
thresh = 3; % running threshold
exclude_calcarine_recordings = true;
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
    opts.save = save_output_for_modeling;
    opts.prewin = -0.05;
    opts.postwin = 0.05;
    opts.spike_rate_thresh = 1;
    opts.normalization = 'zscore'; %'minmax';

    for isess = 1:nsess
        fprintf(1,'%d/%d session\n', isess, nsess);
        RunCorr.(subject){isess} = running_vs_spikePC(D, sessnums(isess), opts);
    end
end

%% get null distribution of correlation using Ken Harris method
NullDist = struct();
for isubj = 1:nsubjs
    subject = subjects{isubj};
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = find(nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2));
    if strcmp(subject, 'marmoset') && exclude_calcarine_recordings
        sessix = setdiff(sessix, [13, 15]);
    end
    nsession = numel(sessix);
    pairlist = nchoosek(1:nsession, 2);
    npairs = size(pairlist,1);
    NullDist.(subject) = nan(npairs,1);
    for ipair = 1:npairs
        x = RunCorr.(subject){pairlist(ipair,1)}.runningspeed;
        y = RunCorr.(subject){pairlist(ipair,2)}.rpc(:,1);
        nx = numel(x);
        ny = numel(y);
        n = min(nx, ny);
        NullDist.(subject)(ipair) = corr(x(1:n), y(1:n), 'type', 'Spearman');
    end
end

%% histogram of spike PC correlation with running



rhos = cell(nsubjs,1);

% where the analyese will print to
if exclude_calcarine_recordings
    figappend = '_excludecalcarine';
else
    figappend = '';
end

fid = fopen(sprintf('output/fig_session_corr%s.txt', figappend), 'w');

fprintf(fid, '*************\n*************\n\nRunning sessionwise analyses with running threshold of %d cm/s\n\n', thresh);
fprintf(fid, 'Histograms of Spearman rank correlation between running and PC 1\n\n');

for isubj = 1:nsubjs
    figure()

    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    npcs = 1;
    
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = find(nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2));
    if strcmp(subject, 'marmoset') && exclude_calcarine_recordings
        sessix = setdiff(sessix, [13, 15]);
    end
    
    for ipc = 1:npcs
        rho = cellfun(@(x) x.rho(ipc), RunCorr.(subject)(sessix));

        % get pvalue by comparing to the null distribution
        pval = arrayfun(@(x) mean(x > NullDist.(subject)), rho);
        pval(pval > .5) = 1 - pval(pval > .5);
        pval = pval * 2; % two-sided

%         pval = cellfun(@(x) x.pval(ipc), RunCorr.(subject)(sessix));

        rhos{isubj} = rho;

        subplot(1, npcs, ipc)

        histogram(rho, 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
        histogram(rho(pval < 0.05), 'binEdges', linspace(-1, 1, 30), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
        plot(median(rho), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2)
        xlim([-1 1])
        
        [p, ~, pstat] = signrank(rho);
        
        fprintf(fid, '%s, median (across sessions) =%02.3f, pval=%d (rank=%02.5f)\n\n', subject, median(rho),p,pstat.signedrank);
        
        plot.offsetAxes(gca)
        ylabel('# sessions')
%         text(.2, .7*max(ylim), sprintf('%02.3f (p = %d, %d, %d)', median(rho), p, pstat.signedrank, numel(rho)), 'FontSize',6, 'FontName', 'Helvetica')
    end
    xlabel("Correlation w/ Running (Spearman's \rho)")
    plot.formatFig(gcf, [2 1.7665], 'jnsci')
    if p < 0.05
        text(median(rho), 1.05*max(ylim), '*', 'FontSize', 8)
    end
    saveas(gcf,fullfile(figdir, sprintf('runpc_corr_hist_%s_%s.pdf', subject, figappend)))
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

% histogram of fraction running
figure(1); clf

% subjects = {'allen', 'gru', 'brie'};
fracsrun = cell(nsubjs,1);
fprintf(fid, 'Histograms of "fraction running" on each session\n\n');

for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = find(nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2));
    if strcmp(subject, 'marmoset') && exclude_calcarine_recordings
        sessix = setdiff(sessix, [13, 15]);
    end
           
    subplot(nsubjs, 1, isubj)
    histogram(fracrun, 'binEdges', linspace(0, 1, 20), 'FaceColor', cmap(2,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
    histogram(fracrun(sessix), 'binEdges', linspace(0, 1, 20), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
    plot(median(fracrun(sessix)), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2)
    xlim([0 1])
    fracsrun{isubj} = fracrun;

    ci = bootci(nboot, @median, fracrun(sessix));
    fprintf(fid,'%s, median (across sessions)=%02.3f, [%02.3f, %02.3f]\n', subject, median(fracrun(sessix)),ci(1), ci(2));

    plot.offsetAxes(gca)
    ylabel('# sessions')
%     text(.2, .7*max(ylim), sprintf('%02.3f [%02.3f, %02.3f]', median(fracrun(sessix)), ci(1), ci(2)), 'FontSize',6, 'FontName', 'Helvetica')
end

xlabel('Fraction Running')
if nsubjs==3
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, sprintf('frac_running_hist%s.pdf', figappend)))
else
    plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
    saveas(gcf,fullfile(figdir, sprintf('frac_running_hist_marm%s.pdf', figappend)))
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




% histogram of running decoding
figure(1); clf
r2_all = [];
fracrun_all = [];
subj_id = [];

fprintf(fid, 'Histograms of running decoding across sessions\n\n');
% get bins for plotting
plot_bins = linspace(-.5, 1, 20);
for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject,false);
    nt = cellfun(@(x) numel(x.runningspeed), RunCorr.(subject));
    fracrun = cellfun(@(x) mean(x.runningspeed > thresh), RunCorr.(subject));
    sessix = find(nt > trial_thresh & fracrun > frac_run_thresh(1) & fracrun < frac_run_thresh(2));
    nsesstot = numel(nt);
    if strcmp(subject, 'marmoset') && exclude_calcarine_recordings
        sessix = setdiff(sessix, [13, 15]);
        nsesstot = nsesstot - 2;
    end
    fprintf(fid, '%s has %d / %d sessions included\n', subject, numel(sessix), nsesstot);

    r2 = cellfun(@(x) x.decoding_r2, RunCorr.(subject)(sessix));
            
    subplot(nsubjs, npcs, (isubj-1)*npcs + 1)
    histogram(r2, 'binEdges', plot_bins, 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
    plot(median(r2), max(ylim)*1.1, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2)
    xlim(plot_bins([1 end]))
    
    [p, ~, pstat] = signrank(r2);
    fprintf(fid,'%s, median=%02.3f, pval=%d (rank=%02.5f)\n', subject, median(r2),p,pstat.signedrank);
    
    plot.offsetAxes(gca)
    ylabel('# sessions')
%     text(.2, .7*max(ylim), sprintf('%02.3f (p = %d, %d, %d)', median(r2), p, pstat.signedrank, numel(r2)), 'FontSize',6, 'FontName', 'Helvetica')
    if p < 0.05
        text(median(r2), 1.05*max(ylim), '*', 'FontSize', 12)
    end

    r2_all = [r2_all; r2(:)];
    fracrun_all = [fracrun_all; fracrun(sessix)];
    subj_id = [subj_id; ones(numel(sessix),1)*isubj];

end

xlabel('Decoding Performance (r^2)')

plot.formatFig(gcf, [1.37 nsubjs*1.21], 'nature')
saveas(gcf,fullfile(figdir, sprintf('rundecoding_hist%s.pdf', figappend)))



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
    saveas(gcf,fullfile(figdir, sprintf('runfrac_vs_decoding%s.pdf', figappend)))
else
    saveas(gcf,fullfile(figdir, sprintf('runfrac_vs_decoding_marm%s.pdf', figappend)))
end

[pval, ~, stats] = ranksum(r2_all(subj_id==1), r2_all(subj_id==2));
fprintf(fid,'\nTesting whether %s and %s Decoding Performance come from distributions with different medians\n', subjects{:});
if pval < 0.05
    fprintf(fid,'%s and %s significantly differ\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );

else
    fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );
end

fclose(fid);

%% plot individual sessions

sort_by_running = false;
sort_by_decoding = false;

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
%         robs = robs - min(robs())
        zz = (robs - min(robs)) ./ range(robs);
        imagesc(zz'); hold on
%         title(['$\rho:$ ' num2str(RunCorr.(subject){isess}.rho(1), 3), ',   $r^2:$' num2str(RunCorr.(subject){isess}.decoding_r2, 3)], 'interpreter', 'latex')
%         nt = max(nt, trial_thresh);
        nt = min(nt, 800);
%         nt = nt + 20;

        xlim([1 nt])
        axis off
        hold on
        plot([nt nt], size(robs,2)-[0 10], 'k', 'Linewidth', 1)
        text(nt+5, size(robs,2), sprintf('%d neurons', 10), 'FontSize',6, 'FontName', 'Helvetica', 'Rotation', 90);
        colormap(cmap)

        axes('Position', [.1 .2 .8 .2])
%         plot(runspeed, 'Color', repmat(.65, 1, 3)); hold on
%         beta = lsqcurvefit(@(beta, x) beta(1)^2*x + beta(2), [max(runspeed) min(runspeed)], rpc(:,1), runspeed, [], [], struct('Display', 'off'));
        xx = (rpc(:,1) - min(rpc(:,1))) / range(rpc(:,1));

%         plot(rpc(:,1)*beta(1)^2 + beta(2), 'Color', cmap(6,:));
        plot(xx * range(runspeed) + min(runspeed),'Color', cmap(6,:), 'Linewidth', .6);
        xlim([1 nt])
        text(10, 1.05*max(runspeed), ['$\rho:$ ' num2str(RunCorr.(subject){isess}.rho(1), 3)], 'interpreter', 'latex')
        ylim(ylim().*[1 1.25])
        axis off
    
        axes('Position', [.1 .08 .8 .15])
        plot(15+runspeed, 'Color', repmat(.65, 1, 3), 'Linewidth', .6); hold on
%         plot(15+runhat, 'Color', cmap(6,:))
%         text(10, .7*max(ylim),['$r^2:$' num2str(RunCorr.(subject){isess}.decoding_r2, 3)], 'interpreter', 'latex')

        hold on
        xlim([1 nt])
        plot([1 51], 0*[1 1], 'k', 'Linewidth', 1)
        text(5, -20, sprintf('%d trials', 50), 'FontSize',6, 'FontName', 'Helvetica')
        mxsp = nanmax(runspeed)/2;
        mxsp = roundn(mxsp, 1);

        plot([1 1]*(nt), 10+[0 mxsp], 'k', 'Linewidth', 1)
        text(nt+5, 0, sprintf('%d cm/s', mxsp), 'FontSize',6, 'FontName', 'Helvetica', 'Rotation', 90);
        axis off
        
        plot.formatFig(gcf, [3.93 1.2], 'nature')
        export_fig(gcf, fullfile(figdir, sprintf('runpc_%s_%d_%d.pdf', subject, i, isess)), '-pdf', '-transparent', '-preserve_size', '-nocrop')
%         saveas(gcf,fullfile(figdir, sprintf('runpc_%s_%d_%d.pdf', subject, i, isess)))


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
        saveas(gcf,fullfile(figdir, sprintf('rundecoding_%s_rank%d_id%d.pdf', subject, i, isess)))


%         iix = runspeed >= 0 & runhat >=0;
%         clf
%         h = qqplot(runspeed(iix), runhat(iix));
%         h(1).MarkerEdgeColor = cmap(2,:);
%         h(1).Marker = 'o';
%         h(1).MarkerSize = 2;
%         h(1).MarkerFaceColor = cmap(6,:);
%         h(2).Color = 'k';
%         h(3).Color = 'k';
%         plot.formatFig(gcf, [1.2 1.2], 'nature')
%         saveas(gcf,fullfile(figdir, sprintf('rundecodingqqplot_%s_rank%d_id%d.pdf', subject, i, isess)))

    end
end


close all

disp('Done with fig_session_corr.m')

%% analyze correlation with pupil
isubj = 2;
subject = subjects{isubj};
rho_pupil = cell2mat(cellfun(@(x) x.rho_pupil', RunCorr.(subject), 'uni', 0));
rho_run = cell2mat(cellfun(@(x) x.rho', RunCorr.(subject), 'uni', 0));

figure(isubj); clf
[~, inds] = sort(rho_run(:,1));

subplot(2,1,1)
plot(rho_run(inds,:), '-o'); hold on
plot(xlim, [0 0], 'k')
ylim([-1 1])
title('running')
subplot(2,1,2)
plot(rho_pupil(inds,:), 'o-'); hold on
plot(xlim, [0 0], 'k--')
ylim([-1 1])
title('pupil')
xlabel('Session #')

plot.suplabel(subject, 't')
