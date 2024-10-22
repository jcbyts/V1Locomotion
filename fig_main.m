
%% paths
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';
addpath ../export_fig
subjects = {'mouse', 'marmoset'};
nsubjs = numel(subjects);

if ~exist('output', 'dir')
    mkdir('output');
end

%% Do main analysis
Stat = struct();
if exist('output/MainAnalysisUnweighted.mat', 'file')
    Stat = load('output/MainAnalysisUnweighted.mat');
else
    for isubj = 1:nsubjs
        subject = subjects{isubj};
        D = load_subject(subject, fpath);

        Stat.(subject) = do_spike_count_analyses(D);
    end

    if Stat.(subject).opts.weighted_spike_count
        save('output/MainAnalysisWeighted.mat', '-v7.3', '-struct', 'Stat');
    else
        save('output/MainAnalysisUnweighted.mat', '-v7.3', '-struct', 'Stat');
    end
end

%% Sort units by foveal / peripheral recordings
% subject = 'marmoset';
% D = load_subject(subject, fpath);
% 
% sessnums = unique(D.sessNumGratings);
% NC = numel(unique(D.spikeIds));
% assert(NC==numel(Stat.(subject).cid))
% Stat.(subject).sessecc = nan(NC, 1);
% Stat.(subject).sessid = nan(NC, 1);
% for id = sessnums(:)'
%     unitIds = unique(D.spikeIds(D.sessNumSpikes==id));
%     rfs = D.units(unitIds);
%     rfs = rfs(~cellfun(@isempty, rfs));
%     
%     ecc = cellfun(@(x) hypot(x{1}.center(1), x{1}.center(2)), rfs);
%     bins = linspace(0, 30, 30);
%     values = histcounts(ecc, bins);
%     w = values.^5 / sum(values.^5);
%     m = bins(1:end-1)*w';
%     fprintf('%d) %d units (rf ecc = %.3f)\n', id, numel(unique(D.spikeIds(D.sessNumSpikes==id))), m)
%     iix = find(ismember(Stat.(subject).cid, unitIds));
%     Stat.(subject).sessecc(iix) = m;
%     Stat.(subject).sessid(iix) = id;
% end
% 
% % %%
% cids = unique(D.spikeIds);
% 
% n = numel(cids);
% sesslist = cell(n,1);
% for cc = 1:n
%     sesslist{cc} = unique(D.sessNumSpikes(D.spikeIds==cids(cc)));
%     if any(ismember([13, 15], sesslist{cc}))
%         fprintf('%d (%d): %d\n', cc, cids(cc), sesslist{cc})
%     end
% end
% %%
% sessid = 13;
% sessix = ~cellfun(@isempty,Stat.marmoset.robs) & ismember(Stat.marmoset.sessid,sessid);
% x = (Stat.marmoset.robs(sessix));
% r = Stat.marmoset.runningspeed(find(sessix, 1));
% figure,
% robs = cell2mat(cellfun(@(x) x', x, 'uni', 0))';
% robs = (robs - min(robs)) ./ range(robs);
% imagesc(robs');
% hold on
% plot(r{1}*1, 'r')
% 
% runix = r{1} > 3;
% statix = abs(r{1}) < 3;
% figure
% plot(mean(robs(statix,:)), mean(robs(runix,:)), '.'); hold on
% plot(Stat.marmoset.frStimS(sessix), Stat.marmoset.frStimR(sessix), 'o')
% plot(xlim, xlim, 'k')
% 
% clf
% plot(mean(robs(statix,:)),Stat.marmoset.frStimS(sessix), '.')
%% Plot Unit Selectivity and Examples
field = 'rateweight';
nboot = 500;
osis = cell(nsubjs, 1);
exclude_calcarine = false;
fig_visible = 'off';

if exclude_calcarine
    fileappend = '_exclude_calcarine';
else
    fileappend = ''; %#ok<*UNRCH> 
end

fid = fopen(sprintf("output/unitselectivity_%s%s.txt", field, fileappend), 'w');
clrs = cell(nsubjs,1);

for isubj = 1:nsubjs
    subject = subjects{isubj};
    cmap = getcolormap(subject, false);
    clrs{isubj} = cmap;
    fprintf(fid, '\n***************************\n');
    fprintf(fid, '***************************\n');
    fprintf(fid, '%s Checking Neuron Selectivity\n', subject);

    NC = numel(Stat.(subject).meanrate);
    good = find(~isnan(Stat.(subject).meanrate));
    if exclude_calcarine
        fprintf(fid, 'Excluding calcarine sessions\n');
        if strcmp(subject, 'marmoset')
            good = intersect(good, find(~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid)));
        end
    else
        fprintf(fid, 'calcarine sessions ARE included\n');
    end

    dsi = nan(NC,1);
    osi = nan(NC,1);
    maxrate = nan(NC,1);
    brate = nan(NC, 2);

    for cc = good(:)'
        baseline = Stat.(subject).baselinerate(cc);
        
        % OSI
        mu = squeeze(Stat.(subject).(['d' field])(cc,:,2));

        brate(cc,:) = [baseline min(mu)];

        maxrate(cc) = max(mu);

        dsi(cc) = direction_selectivity_index(Stat.(subject).directions, mu(:)-baseline, 2);
        osi(cc) = orientation_selectivity_index(Stat.(subject).directions, mu(:)-baseline, 2);
    end

    % remove nans
    good(isnan(osi(good))) = [];

    % find units with rates that are suppressed during gratings
    suppressed = brate(:,1) > brate(:,2);

    figure('Visible', fig_visible)

    iix = intersect(find(Stat.(subject).pvis(:,3)<0.05), good);
    fprintf(fid, '%d units\n', numel(good));
    fprintf(fid, '%d visually driven\n', numel(iix));
    fprintf(fid, '%d suppressed\n', sum(suppressed(iix)));

    iix((suppressed(iix)))= [];
    fprintf(fid, '%d make it in the plot\n', numel(iix));
    mosi = median(osi(iix));
    osis{isubj} = osi(iix);
    ci = bootci(nboot, @median, osi(iix));
    fprintf(fid, 'median OSI: %02.2f [%02.2f, %02.2f]\n', mosi, ci(1), ci(2));

    histogram(osi(iix), 'binEdges', linspace(0,1,50), 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1); hold on
    plot(mosi, 1.05*max(ylim), 'v', 'MarkerFaceColor', cmap(6,:), 'Color', 'none', 'MarkerSize', 2)
    xlabel('OSI')
    set(gca, 'XTick', 0:.25:1)
    set(gcf, 'Position', [0 0 300*[1.2 1]])
    plot.formatFig(gcf, [1.2 1], 'jnsci')
    set(gca, 'Color', 'none')
    export_fig(fullfile(figdir, sprintf('osi_dist_%s.pdf', subject)), '-transparent', '-pdf')

    fontargs = {'FontName', 'Helvetica', 'FontSize', 7, 'FontWeight', 'normal'};
    metric = osi(:);
    
    iix = iix(brate(iix,1) < 15); % exclude multi-units by firing rate
    [~, inds] = min((metric(iix) - [.1 .55 .9]).^2);
    inds = iix(inds);

    for i = 1:3
        cc = inds(i);
        subplot(1,3,i)
        mu = squeeze(Stat.(subject).(['d' field])(cc,:,:));
        baseline = Stat.(subject).baselinerate(cc);

        plot(Stat.(subject).directions, mu, 'Color', cmap(6,:))
        hold on
        plot(Stat.(subject).directions, mu(:,2), '.', 'Color', cmap(6,:))
        osi0 = orientation_selectivity_index(Stat.(subject).directions, mu(:,2)' - baseline, 2);
        plot(xlim, baseline*[1 1], '--', 'Color', cmap(4,:))
        title(sprintf('OSI: %02.2f', metric(cc)), fontargs{:})

        plot.offsetAxes(gca, false, 10)
        ylim([0 max(ylim)])
        set(gca, 'XTick', 0:45:330)
        xlim(Stat.(subject).directions([1 end]))
        xlabel('Orientation (deg.)', fontargs{:})
        if i==1
            ylabel('Firing Rate (sp s^{-1})', fontargs{:})
        end
    end
    
    plot.formatFig(gcf, [3 1], 'jnsci')
    set(gca, 'Color', 'none')
    set(gcf, 'Color', 'none')
    export_fig(gcf, fullfile(figdir, sprintf('example_tuning_%s.pdf', subject)), '-transparent', '-pdf', '-native')
%     print(gcf, fullfile(figdir, sprintf('example_tuning_%s.pdf', subject)), '-dpdf')

end


[pval, ~, stats] = ranksum(osis{:});
fprintf(fid,'Testing whether %s and %s OSI come from distributions with different medians\n', subjects{:});
if pval < 0.05
    fprintf(fid,'%s and %s significantly differ\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%02.7f, ranksum=%d\n', pval, stats.ranksum );

else
    fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
end

fclose(fid);



%% Plot scatter hists
trial_thresh = 250;
tuningfield = 'rateweight';
yscale = 'log';
ms = 3; % marker size

unit_groups = {'all', 'tuned', 'vis', 'suppressed', 'foveal', 'peripheral', 'npx_calc', 'npx_fov', 'npx_calc_tuned', 'npx_fov_tuned'};
frDiffFovealPeripheralStim = cell(2,2);
frDiffFovealPeripheralPref = cell(2,2);
frDiffNpxStim = cell(2,2);
frDiffNpxPref = cell(2,2);

for igroup = 1:numel(unit_groups)

    plot_index = unit_groups{igroup}; %'foveal'; % tuned, vis, suppressed, all, foveal, peripheral

    fid = fopen(sprintf("output/main_analysis_%s_%s%s.txt", tuningfield, plot_index, fileappend), 'w');
    fprintf(fid, '***************************\n');
    fprintf(fid, '***************************\n');
    fprintf(fid, '***************************\n\n');
    fprintf(fid, "Running statistics on the spike rate scatter plots\nwith the following conditions:\n");
    fprintf(fid, "nboot: %d\ntrial_thresh: %d\ntuningfield: %s\nunit_index: %s\n", nboot, trial_thresh, tuningfield, plot_index);

    fprintf(fid, 'Note: "tuning field" refers to how the preferred stimulus rate was calculated\n');
    fprintf(fid, '\t"ratemarg" means it is the firing rate in the preferred direction after marginalizing over SF and TF\n');
    fprintf(fid, '\t"ratebest" means it is the firing rate in the preferred direction at the best SF and TF\n');
    fprintf(fid, '\t"rateweight" means it is the firing rate in the preferred direction with weighted averging over SF and TF by the tuning\n');

    % subjects = fieldnames(Stat);
    nsubjs = numel(subjects);
    frDiffBase = cell(nsubjs, 2); % raw difference and log(ratio)
    frDiffStim = cell(nsubjs, 2);
    frDiffPref = cell(nsubjs, 2);

    for isubj = 1:nsubjs
        subject = subjects{isubj};
        cmap = getcolormap(subject, false);

        fprintf(fid, '***************************\n\n');
        fprintf(fid, '%s\n', subject);
        fprintf(fid, '***************************\n\n');

        % firing rate during stimulus (regardless of direction)
        frStimR = Stat.(subject).frStimR;
        frStimS = Stat.(subject).frStimS;

        % --- compute some indices
        good = ~isnan(Stat.(subject).meanrate); % units with > 1 spike / sec
        numtrials = cellfun(@numel, Stat.(subject).robs);
        good = good & numtrials > trial_thresh;

        NC = numel(good);
        osi = nan(NC,1);
        brate = nan(NC, 2);
        pvis = Stat.(subject).pvis(:,3)<0.05;
        frPrefR = nan(NC,3);
        frPrefS = nan(NC,3);

            ttestp = nan(NC,1);
            ttestci = nan(NC,2);

        for cc = find(good(:))'
            baseline = Stat.(subject).baselinerate(cc);

            % DSI
            mu = squeeze(Stat.(subject).(['d' tuningfield])(cc,:,2));
            [~,prefid] = max(mu);
            frPrefR(cc,:) = Stat.(subject).(['d' tuningfield 'R'])(cc,prefid,:);
            frPrefS(cc,:) = Stat.(subject).(['d' tuningfield 'S'])(cc,prefid,:);

            brate(cc,:) = [baseline min(mu)];

            baseline = min(baseline, min(mu));

            osi(cc) = orientation_selectivity_index(Stat.(subject).directions, mu(:)-baseline, 2);

            % ttest on running
            r = Stat.(subject).robs{cc};
            statix = abs(Stat.(subject).runningspeed{cc} ) < 3;
            runix = (Stat.(subject).runningspeed{cc} ) > 3;
            [H,P,CI] = ttest2(r(statix), r(runix),'Vartype','unequal');
            ttestp(cc) = P;
            ttestci(cc,:) = CI;

        end



        good = ~isnan(Stat.(subject).meanrate);
        rfecc = Stat.(subject).rfecc;

        switch plot_index
            case 'vis'
                good = good & pvis;
                if exclude_calcarine && strcmp(subject, 'marmoset')
                    good = good & ~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
            case 'tuned'
                good = good & pvis & osi > .2;
                if exclude_calcarine && strcmp(subject, 'marmoset')
                    good = good & ~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
            case 'suppressed'
                good = good & brate(:,1) > brate(:,2);
                if exclude_calcarine && strcmp(subject, 'marmoset')
                    good = good & ~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
            case 'foveal'
                if strcmp(subject, 'marmoset')
                    good = good & rfecc < 3;
                end
                if exclude_calcarine && strcmp(subject, 'marmoset')
                    good = good & ~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
            case 'peripheral'
                if strcmp(subject, 'marmoset')
                    good = good & rfecc > 3;
                end
                if exclude_calcarine && strcmp(subject, 'marmoset')
                    good = good & ~cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
            case 'npx_calc'
                if strcmp(subject, 'marmoset')
                   good = good & cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end

            case 'npx_fov'
                if strcmp(subject, 'marmoset')
                   good = good & cellfun(@(x) any(ismember(x, [14 16])), Stat.(subject).sessid);
                end
            case 'npx_calc_tuned'
                if strcmp(subject, 'marmoset')
                    good = good & cellfun(@(x) any(ismember(x, [13 15])), Stat.(subject).sessid);
                end
                good = good & osi > .2;

            case 'npx_fov_tuned'
                if strcmp(subject, 'marmoset')
                    good = good & cellfun(@(x) any(ismember(x, [14 16])), Stat.(subject).sessid);
                end
                good = good & osi > .2;
        end

        fprintf(fid, 'Using [%s] units only (%d units)\n', plot_index, sum(good));
        if sum(good)==0
            continue
        end

        frStimR = frStimR(good,:);
        frStimS = frStimS(good,:);
        frPrefR = frPrefR(good,:);
        frPrefS = frPrefS(good,:);

        NC = size(frStimR,1);

        decStimIx = find(Stat.(subject).bootTestMeanfrStim(good) < .025);
        incStimIx = find(Stat.(subject).bootTestMeanfrStim(good) > .975);
        notSigStimIx = setdiff( (1:NC)', [incStimIx; decStimIx]);

        % Preferred direction FIRING RATE (marginalized across all conditions)
        decPrefIx = find(Stat.(subject).prctilerunbootbest(good) < .025);
        incPrefIx = find(Stat.(subject).prctilerunbootbest(good) > .975);
        notSigPrefIx = setdiff( (1:NC)', [incPrefIx; decPrefIx]);

        % STIM-DRIVEN FIRING RATE (marginalized across all conditions)
%         figure(isubj*10+4); clf
%         set(gcf, 'Visible', fig_visible)
        figure('Visible', fig_visible)
        plot(frStimS(:,[2 2])', frStimR(:,[1 3])', 'Color', .5*[1 1 1 .1]); hold on
        plot(frStimS(:,[1 3])', frStimR(:,[2 2])', 'Color', .5*[1 1 1 .1])

        plot(frStimS(notSigStimIx,2), frStimR(notSigStimIx,2), 'o', 'Color', [1 1 1], 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', [cmap(4,:)])
        plot(frStimS(incStimIx,2), frStimR(incStimIx,2), 'o', 'Color', [1 1 1], 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', cmap(6,:))
        plot(frStimS(decStimIx,2), frStimR(decStimIx,2), 'o', 'Color', [1 1 1], 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', cmap(6,:))

        xlabel('Stationary Firing Rate')
        ylabel('Running Firing Rate')
%         title('Stim-driven firing rate')

        if strcmp(yscale, 'log')
            xlim([1 100])
            ylim([1 100])
            l = plot([1 100], [1 100]);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = plot([1 100], [1 100]*2);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = plot( 1:100, (1:100)/2);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            xlim([1 100])
            ylim([1 100])
        else
            xlim([0 100])
            ylim([0 100])
            l = plot([0 100], [0 100]);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = refline(1,20);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = refline(1,-20);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            xlim([0 100])
            ylim([0 100])
        end

        axis square
        plot.formatFig(gcf, [1.96 1.96], 'jnsci')
        set(gca, 'Yscale', yscale, 'Xscale', yscale)
        export_fig(gcf, fullfile(figdir, sprintf('rate_compare_stim_%s_%s_%s.pdf', yscale, plot_index, subject)), '-pdf', '-transparent')

%         figure(isubj*10+2); clf
%         set(gcf, 'Visible', fig_visible)
        figure('Visible', fig_visible)
        plot(frPrefS(:,[2 2])', frPrefR(:,[1 3])', 'Color', .5*[1 1 1 .1]); hold on
        plot(frPrefS(:,[1 3])', frPrefR(:,[2 2])', 'Color', .5*[1 1 1 .1])

        plot(frPrefS(notSigPrefIx,2), frPrefR(notSigPrefIx,2), 'o', 'Color', 'w', 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', cmap(4,:))
        plot(frPrefS(incPrefIx,2), frPrefR(incPrefIx,2), 'o', 'Color', 'w', 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', cmap(6,:)); hold on
        plot(frPrefS(decPrefIx,2), frPrefR(decPrefIx,2), 'o', 'Color', 'w', 'Linewidth', .25, 'MarkerSize', ms, 'MarkerFaceColor', cmap(6,:))


        xlabel('Stationary Firing Rate')
        ylabel('Running Firing Rate')
%         title('Preferred Stim firing rate')

        if strcmp(yscale, 'log')
            xlim([1 100])
            ylim([1 100])
            l = plot([1 100], [1 100]);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = plot([1 100], [1 100]*2);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = plot( 1:100, (1:100)/2);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            xlim([1 100])
            ylim([1 100])
        else
            xlim([0 100])
            ylim([0 100])
            l = plot([0 100], [0 100]);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = refline(1,20);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            l = refline(1,-20);
            set(l, 'color', 'k', 'linestyle', '--', 'linewidth', 0.5);
            xlim([0 100])
            ylim([0 100])
        end

        axis square
        plot.formatFig(gcf, [1.96 1.96], 'jnsci')
        set(gca, 'Yscale', yscale, 'Xscale', yscale)
        export_fig(gcf, fullfile(figdir, sprintf('rate_compare_pref_%s_%s_%s.pdf', yscale, plot_index, subject)), '-pdf', '-transparent')


        % REPORT STATS
        nIncStim = numel(incStimIx);
        nDecStim = numel(decStimIx);

        modUnits = unique([incStimIx; decStimIx]);
        nModUnits = numel(modUnits);

        fprintf(fid, '%d/%d (%02.2f%%) increased stim firing rate\n', nIncStim, NC, 100*nIncStim/NC);
        fprintf(fid, '%d/%d (%02.2f%%) decreased stim firing rate\n', nDecStim, NC, 100*nDecStim/NC);

        fprintf(fid, '%d/%d (%02.2f%%) total modulated units\n', nModUnits, NC, 100*nModUnits/NC);

        fprintf(fid, '\nWilcoxon signed rank test for running modulation:\n');
        [pvalStim, ~, sStim] = signrank(frStimS(:,2), frStimR(:,2));
        mS = median(frStimS(:,2));
        ciS = bootci(nboot, @median, frStimS(:,2));
        mR = median(frStimR(:,2));
        ciR = bootci(nboot, @median, frStimR(:,2));

        fprintf(fid, '[All Stimuli] condition\n');
        fprintf(fid, 'Medians for running %02.2f [%02.2f, %02.2f] and stationary %02.2f [%02.2f, %02.2f]\n', mR, ciR(1), ciR(2), mS, ciS(1), ciS(2));
        if pvalStim < 0.05
            fprintf(fid, 'Are significantly different\n');
        else
            fprintf(fid, 'Are NOT significantly different\n');
        end
        fprintf(fid, 'p = %d, signedrank=%d\n', pvalStim, sStim.signedrank);

        [pvalPref, ~, sPref] = signrank(frPrefS(:,2), frPrefR(:,2));
        mS = nanmedian(frPrefS(:,2)); %#ok<*NANMEDIAN>
        ciS = bootci(nboot, @nanmedian, frPrefS(~isnan(frPrefR(:,2)),2));
        mR = nanmedian(frPrefR(:,2));
        ciR = bootci(nboot, @nanmedian, frPrefR(~isnan(frPrefR(:,2)),2));

        fprintf(fid, '[Preferred Stimulus] condition\n');
        fprintf(fid, 'Medians for running %02.2f [%02.2f, %02.2f] and stationary %02.2f [%02.2f, %02.2f]\n', mR, ciR(1), ciR(2), mS, ciS(1), ciS(2));
        if pvalPref < 0.05
            fprintf(fid, 'Are significantly different\n');
        else
            fprintf(fid, 'Are NOT significantly different\n');
        end
        fprintf(fid, 'p = %d, signedrank=%d\n', pvalPref, sPref.signedrank);

        % across subject / condition
        frDiffStim{isubj,1} = frStimR(:,2) - frStimS(:,2);
        frDiffPref{isubj,1} = frPrefR(:,2) - frPrefS(:,2);

        frDiffStim{isubj,2} = log10(frStimR(:,2)) - log10(frStimS(:,2));
        frDiffPref{isubj,2} = log10(frPrefR(:,2)) - log10(frPrefS(:,2));

        if strcmp(subject, 'marmoset')
            switch plot_index
                case 'foveal'
                    frDiffFovealPeripheralStim{1,1} = frDiffStim{isubj,1};
                    frDiffFovealPeripheralStim{1,2} = frDiffStim{isubj,2};

                    frDiffFovealPeripheralPref{1,1} = frDiffPref{isubj,1};
                    frDiffFovealPeripheralPref{1,2} = frDiffPref{isubj,2};

                case 'peripheral'
                    frDiffFovealPeripheralStim{2,1} = frDiffStim{isubj,1};
                    frDiffFovealPeripheralStim{2,2} = frDiffStim{isubj,2};

                    frDiffFovealPeripheralPref{2,1} = frDiffPref{isubj,1};
                    frDiffFovealPeripheralPref{2,2} = frDiffPref{isubj,2};

                case 'npx_calc'
                    frDiffNpxStim{2,1} = frDiffStim{isubj,1};
                    frDiffNpxStim{2,2} = frDiffStim{isubj,2};

                    frDiffNpxPref{2,1} = frDiffPref{isubj,1};
                    frDiffNpxPref{2,2} = frDiffPref{isubj,2};

                case 'npx_fov'
                    frDiffNpxStim{1,1} = frDiffStim{isubj,1};
                    frDiffNpxStim{1,2} = frDiffStim{isubj,2};

                    frDiffNpxPref{1,1} = frDiffPref{isubj,1};
                    frDiffNpxPref{1,2} = frDiffPref{isubj,2};


            end
        end

        rrat = frStimR(:,2)./frStimS(:,2);
        good = ~(isnan(rrat) | isinf(rrat) | rrat==0);
        [m, ci] = geomeanci(rrat(good));

        fprintf(fid, "geometric mean stim-driven firing rate ratio (Running:Stationary) is %02.3f [%02.3f, %02.3f] (n=%d)\n", m, ci(1), ci(2), NC);

        rrat = frPrefR(:,2)./frPrefS(:,2);
        good = ~(isnan(rrat) | isinf(rrat) | rrat==0);
        [m, ci] = geomeanci(rrat(good));

        fprintf(fid, "geometric mean pref-stim firing rate ratio (Running:Stationary) is %02.3f [%02.3f, %02.3f] (n=%d)\n", m, ci(1), ci(2), NC);


        % STIM-DRIVEN
        if strcmp(yscale, 'log')
%             figure(isubj*10+2, 'Visible', fig_visible); clf
            figure('Visible', fig_visible)
            bins = linspace(-1,1,100);
            histogram(log10(frStimS(:,2))-log10(frStimR(:,2)), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(4,:), 'FaceAlpha', 1); hold on
            issig = modUnits; %frStimR(:,2) < frStimS(:,1) | frStimR(:,2) > frStimS(:,3);
            histogram(log10(frStimS(issig,2))-log10(frStimR(issig,2)), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(6,:), 'FaceAlpha', 1);
            mddiff = nanmedian(log10(frStimS(:,2))-log10(frStimR(:,2)));
            plot(mddiff, 1.05*max(ylim), 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2)
            plot([0 0], ylim, 'k')
            plot(log10(2)*[1 1], ylim, 'k--')
            plot(-log10(2)*[1 1], ylim, 'k--')
%             plot.formatFig(gcf, [1.73 .669], 'jnsci')
            plot.formatFig(gcf, [1.45 0.5608], 'jnsci')
            set(gca, 'XTick', [])
            export_fig(gcf, fullfile(figdir, sprintf('rate_compare_stimhist_log_%s_%s.pdf', plot_index, subject)), '-pdf', '-transparent')
        else
%             figure(isubj*10+2); clf
%             set(gcf, 'Visible', fig_visible)
            figure('Visible', fig_visible)
            histogram(frStimS(:,2)-frStimR(:,2), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(4,:), 'FaceAlpha', 1); hold on
            issig = modUnits; %frStimR(:,2) < frStimS(:,1) | frStimR(:,2) > frStimS(:,3);
            histogram(frStimS(issig,2)-frStimR(issig,2), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(6,:), 'FaceAlpha', 1);
            plot([0 0], ylim, 'k')
            plot(20*[1 1], ylim, 'k--')
            plot(-20*[1 1], ylim, 'k--')
%             plot.formatFig(gcf, [1.73 .669], 'jnsci')
            plot.formatFig(gcf, [1.45 0.5608], 'jnsci')
            set(gca, 'XTick', [])
            export_fig(gcf, fullfile(figdir, sprintf('rate_compare_stimhist_%s_%s.pdf', plot_index, subject)), '-pdf', '-transparent')
        end


        % PREF-STIM
        if strcmp(yscale, 'log')
%             figure(isubj*10+6); clf
            figure('Visible', fig_visible)
            bins = linspace(-1,1,100);
            histogram(log10(frPrefS(:,2))-log10(frPrefR(:,2)), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(4,:), 'FaceAlpha', 1); hold on
            issig = modUnits; %frPrefR(:,2) < frPrefS(:,1) | frPrefR(:,2) > frPrefS(:,3);
            histogram(log10(frPrefS(issig,2))-log10(frPrefR(issig,2)), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(6,:), 'FaceAlpha', 1);
            mddiff = nanmedian(log10(frPrefS(:,2))-log10(frPrefR(:,2)));
            plot(mddiff, 1.05*max(ylim), 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 2)
            plot([0 0], ylim, 'k')
            plot(log10(2)*[1 1], ylim, 'k--')
            plot(-log10(2)*[1 1], ylim, 'k--')
%             plot.formatFig(gcf, [1.73 .669], 'jnsci')
            plot.formatFig(gcf, [1.45 0.5608], 'jnsci')
            set(gca, 'XTick', [])
            export_fig(gcf, fullfile(figdir, sprintf('rate_compare_prefhist_log_%s_%s.pdf', plot_index, subject)), '-pdf', '-transparent')
        else
            figure('Visible', fig_visible)
%             figure(isubj*10+6); clf
%             set(gcf, 'Visible', fig_visible)
            histogram(frPrefS(:,2)-frPrefR(:,2), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(4,:), 'FaceAlpha', 1); hold on
            issig = modUnits; %frPrefR(:,2) < frPrefS(:,1) | frPrefR(:,2) > frPrefS(:,3);
            histogram(frPrefS(issig,2)-frPrefR(issig,2), 'binEdges', bins, 'EdgeColor', 'none', 'FaceColor', cmap(6,:), 'FaceAlpha', 1);
            plot([0 0], ylim, 'k')
            plot(20*[1 1], ylim, 'k--')
            plot(-20*[1 1], ylim, 'k--')
%             plot.formatFig(gcf, [1.73 .669], 'jnsci')
            plot.formatFig(gcf, [1.45 0.5608], 'jnsci')
            set(gca, 'XTick', [])
            print(gcf, fullfile(figdir, sprintf('rate_compare_prefhist_%s_%s.pdf', plot_index, subject)), '-dpdf')
%             export_fig(gcf, fullfile(figdir, sprintf('rate_compare_prefhist_%s_%s.pdf', plot_index, subject)), '-pdf', '-transparent')
        end

    end

    % --- Compare mouse and marmoset across conditions
    conditions = {'All Stimuli', 'Preferred Stimulus'};
    fprintf(fid, '\n\n***********************\nComparing %s and %s\n', subjects{:});
%     if contains(plot_index, 'foveal') || contains(plot_index, 'npx')
%         fprintf(fid, 'Mouse has no foveal condition. skipping\n');
%     else

    for i = 1:numel(conditions)
        cond = conditions{i};

        switch cond
            case {'All Stimuli'}
                frDiff = frDiffStim;
            case {'Preferred Stimulus'}
                frDiff = frDiffPref;
        end

%         figure(igroup*100 + i); clf
%         set(gcf, 'Visible', fig_visible)
        figure('Visible', fig_visible)
        subplot(1,2,1)
        histogram(frDiff{2,1}, 'binEdges', linspace(-20,20,50), 'Normalization', 'probability', 'EdgeColor', 'none', 'FaceColor', clrs{2}(6,:)); hold on
        histogram(frDiff{1,1}, 'binEdges', linspace(-20,20,50), 'Normalization', 'probability', 'EdgeColor', 'none', 'FaceColor', clrs{1}(6,:));
        xlabel('Rate (run) - Rate (stat)')
        ylabel('Probability')
        title(sprintf("Group: %s", plot_index))

        [pval, ~, stats] = ranksum(frDiff{:,1});

        fprintf(fid, '\n\n***********************\n');
        fprintf(fid, '\n\nCondition: modulation of [%s] spike rate\n\n', cond);
        fprintf(fid, '\nSpike Rate Differences\nTest for different medians\n');
        fprintf(fid, 'Wilcoxon Rank Sum Test\n');
        if pval < 0.05
            fprintf(fid,'%s and %s significantly differ\n', subjects{:});
            fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

        else
            fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
            fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
        end

        subplot(1,2,2)
        histogram(frDiff{2,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none', 'FaceColor', clrs{2}(6,:)); hold on
        histogram(frDiff{1,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none', 'FaceColor', clrs{2}(1,:));
        xlabel('log(Rate (run)) - log(Rate (stat))')
        title(cond)
        plot.formatFig(gcf, [2*1.73 .669], 'jnsci')
        print(gcf, fullfile(figdir, sprintf('rate_compare_%s_%s_in_%s_%s.pdf', subjects{1}, subjects{2}, plot_index, cond)), '-dpdf')

        [pval, ~, stats] = ranksum(frDiff{:,2});

        fprintf(fid,'\nLog Spike Rate Ratio\nTest for different medians\n');
        fprintf(fid,'Wilcoxon Rank Sum Test\n');

        if pval < 0.05
            fprintf(fid,'%s and %s significantly differ\n', subjects{:});
            fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

        else
            fprintf(fid,'%s and %s NOT significantly different\n', subjects{:});
            fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
        end

    end
%     end
    fclose(fid);
end

% check foveal vs. peripheral
fid = fopen(sprintf("output/fovea_vs_periphery_%s.txt", tuningfield), 'w');

fprintf(fid, '\n\n***********************\n***********************\n***********************\n');
fprintf(fid, 'Comparing fovea and periphery for marmoset\n');
for icond = 1:numel(conditions)
    figure('Visible', fig_visible)
%     figure(icond); clf
%     set(gcf, 'Visible', fig_visible)

    cond = conditions{icond};
    switch cond
        case 'All Stimuli'
            frDiff = frDiffFovealPeripheralStim;
        case 'Preferred Stimulus'
            frDiff = frDiffFovealPeripheralPref;
    end

    subplot(1,2,1)
    histogram(frDiff{1,1}, 'binEdges', linspace(-20,20,100), 'Normalization', 'probability', 'EdgeColor', 'none'); hold on
    histogram(frDiff{2,1}, 'binEdges', linspace(-20,20,100), 'Normalization', 'probability', 'EdgeColor', 'none');
    title(cond)

    [pval, ~, stats] = ranksum(frDiff{:,1});
    nboot = 500; 
    
    m1ci = bootci(nboot, @nanmedian, frDiff{1,1});
    m1 = nanmedian(frDiff{1,1});
    m2ci = bootci(nboot, @nanmedian, frDiff{2,1});
    m2 = nanmedian(frDiff{2,1});

    fprintf(fid, '\n\n***********************\n');
    fprintf(fid, '\n\nCondition: modulation of [%s] spike rate\n\n', cond);
    fprintf(fid, '\nSpike Rate Differences\nTest for different medians\n');
    fprintf(fid, 'Wilcoxon Rank Sum Test\n');
    fprintf(fid, 'median (fovea) = %.3f [%.3f, %.3f]\n', m1, m1ci(1), m1ci(2));
    fprintf(fid, 'median (periphery) = %.3f [%.3f, %.3f]\n', m2, m2ci(1), m2ci(2));

    if pval < 0.05
        fprintf(fid,'fovea and periphery significantly differ\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

    else
        fprintf(fid,'fovea and periphery NOT significantly different\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
    end

    subplot(1,2,2)
    histogram(frDiff{1,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none'); hold on
    histogram(frDiff{2,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none');
    title(cond)
    plot.formatFig(gcf, [2*1.73 .669], 'jnsci')
    print(gcf, fullfile(figdir, sprintf('rate_compare_fovea_periphery %s.pdf', cond)), '-dpdf')

    [pval, ~, stats] = ranksum(frDiff{:,2});

    m1ci = bootci(nboot, @nanmedian, frDiff{1,2});
    m1 = nanmedian(frDiff{1,2});
    m2ci = bootci(nboot, @nanmedian, frDiff{2,2});
    m2 = nanmedian(frDiff{2,2});

    fprintf(fid,'\nLog Spike Rate Ratio\nTest for different medians\n');
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid, 'median (fovea) = %.3f [%.3f, %.3f]\n', m1, m1ci(1), m1ci(2));
    fprintf(fid, 'median (periphery) = %.3f [%.3f, %.3f]\n', m2, m2ci(1), m2ci(2));

    if pval < 0.05
        fprintf(fid,'fovea and periphery significantly differ\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

    else
        fprintf(fid,'fovea and periphery NOT significantly different\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
    end
end


fprintf(fid, '\n\n***********************\n***********************\n***********************\n');
fprintf(fid, 'Comparing fovea and calcarine recordings from NPX in marmoset\n');
for icond = 1:numel(conditions)
%     figure(icond); clf
%     set(gcf, 'Visible', fig_visible)
    figure('Visible', fig_visible)

    cond = conditions{icond};
    switch cond
        case 'All Stimuli'
            frDiff = frDiffNpxStim;
        case 'Preferred Stimulus'
            frDiff = frDiffNpxPref;
    end

    subplot(1,2,1)
    histogram(frDiff{1,1}, 'binEdges', linspace(-20,20,100), 'Normalization', 'probability', 'EdgeColor', 'none'); hold on
    histogram(frDiff{2,1}, 'binEdges', linspace(-20,20,100), 'Normalization', 'probability', 'EdgeColor', 'none');
    title(cond)

    m1ci = bootci(nboot, @nanmedian, frDiff{1,1});
    m1 = nanmedian(frDiff{1,1});
    m2ci = bootci(nboot, @nanmedian, frDiff{2,1});
    m2 = nanmedian(frDiff{2,1});

    [pval, ~, stats] = ranksum(frDiff{:,1});

    fprintf(fid, '\n\n***********************\n');
    fprintf(fid, '\n\nCondition: modulation of [%s] spike rate\n\n', cond);
    fprintf(fid, '\nSpike Rate Differences\nTest for different medians\n');
    fprintf(fid, 'Wilcoxon Rank Sum Test\n');
    fprintf(fid, 'median (npx fovea) = %.3f [%.3f, %.3f]\n', m1, m1ci(1), m1ci(2));
    fprintf(fid, 'median (npx calcarine) = %.3f [%.3f, %.3f]\n', m2, m2ci(1), m2ci(2));

    if pval < 0.05
        fprintf(fid,'fovea and periphery significantly differ\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

    else
        fprintf(fid,'fovea and periphery NOT significantly different\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
    end

    subplot(1,2,2)
    histogram(frDiff{1,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none'); hold on
    histogram(frDiff{2,2}, 'binEdges', linspace(-1,1,100), 'Normalization', 'probability', 'EdgeColor', 'none');
    title(cond)
    plot.formatFig(gcf, [2*1.73 .669], 'jnsci')
    
    print(gcf, fullfile(figdir, sprintf('rate_compare_npx_fov_calc %s.pdf', cond)), '-dpdf')

    [pval, ~, stats] = ranksum(frDiff{:,2});

    m1ci = bootci(nboot, @nanmedian, frDiff{1,2});
    m1 = nanmedian(frDiff{1,2});
    m2ci = bootci(nboot, @nanmedian, frDiff{2,2});
    m2 = nanmedian(frDiff{2,2});

    fprintf(fid,'\nLog Spike Rate Ratio\nTest for different medians\n');
    fprintf(fid,'Wilcoxon Rank Sum Test\n');
    fprintf(fid, 'median (npx fovea) = %.3f [%.3f, %.3f]\n', m1, m1ci(1), m1ci(2));
    fprintf(fid, 'median (npx calcarine) = %.3f [%.3f, %.3f]\n', m2, m2ci(1), m2ci(2));

    if pval < 0.05
        fprintf(fid,'fovea and periphery significantly differ\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );

    else
        fprintf(fid,'fovea and periphery NOT significantly different\n');
        fprintf(fid,'p=%d, ranksum=%d\n', pval, stats.ranksum );
    end
end

fclose(fid);
close all
