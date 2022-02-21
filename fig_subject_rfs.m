
%% plot RFs
figdir = 'Figures/HuklabTreadmill/manuscript/';
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
subjs = {'gru', 'brie', 'allen'};

%% plot subset of RFs -- this is what is used in the figure
xd = [-50 50];
yd = [-40 40];
for isubj = 1:numel(subjs)
    figure(isubj); clf

    subj = subjs{isubj};
    cmap = getcolormap(subj, false);

    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));
    NC = numel(unique(D.spikeIds));


    mv = cellfun(@(x) x{1}.maxV, D.units(hasrf));
    iix = mv > 10;

    hasrf = hasrf(iix);

    rng(1)
    hasrf = randsample(hasrf, 50, false);
    nrf = numel(hasrf);

    fprintf('%s plotting %d RFs\n', subj, nrf)
    
    plot(xd, [0 0], 'Color', [0 0 0 .3]);
    hold on
    plot([0 0], yd, 'Color', [0 0 0 .3]);

    for i = 1:nrf
        rf = D.units{hasrf(i)}{1};
        if strcmp(subj, 'allen')
            rf.contour(:,1) = rf.contour(:,1) - 50;
        end
        rf.contour = rf.contour + randn*.2;
        plot(rf.contour(:,1), rf.contour(:,2), 'Color', [cmap(6,:) .5]); hold on
    end

    xlim(xd)
    ylim(yd)

    plot(0, 0, '+k', 'MarkerSize', 2)

    plot(-40*[1 1], -30+[0 20], 'k', 'Linewidth', 2)
    plot(-40+[0 20], -30*[1 1], 'k', 'Linewidth', 2)

    plot.formatFig(gcf, [1.25 1], 'nature', 'OffsetAxesBool', true)
    
    set(gca, 'XTick', [xd(1) 0 xd(2)])
    set(gca, 'YTick', [yd(1) 0 yd(2)])
    xlim(xd)
    ylim(yd)

    axis equal
    saveas(gcf, fullfile(figdir, sprintf('rfs_subset_%s.pdf', subj)))
end

%% plot example RFs and contour
cids = {};
cids{1} = [1 101];
cids{2} = [1 20];

for isubj = 1:2
    subj = subjs{isubj};
    cmap = getcolormap(subj, false);

    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));

    for cc = cids{isubj}

        unitid = hasrf(cc);

        xax = D.units{unitid}{1}.xax;
        yax = D.units{unitid}{1}.yax;
        srf = D.units{unitid}{1}.srf;
        srf = (srf - min(srf(:))) / range(srf(:));
        con = D.units{unitid}{1}.contour;

        figure(cc); clf
        set(gcf, 'Color', 'w')
        imagesc(xax, yax, srf, [0 1]); colormap(plot.coolwarm)
        hold on
        plot(con(:,1), con(:,2), 'k')
        axis xy
        set(gca, 'XTick', -30:10:30, 'YTick', -10:5:10)

        plot.offsetAxes(gca)
        xlim([-30 30])
        ylim([-10 10])
        axis equal
        plot.formatFig(gcf, [1 1], 'nature')
        saveas(gcf, fullfile(figdir, sprintf("rf_examples_%s_%d.pdf", subj, unitid)))
    end
end

%% plot all RF contours

for isubj = 1:numel(subjs)
    figure(isubj); clf

    subj = subjs{isubj};
    cmap = getcolormap(subj, false);

    D = load_subject(subj, fpath);

    hasrf = find(~cellfun(@isempty, D.units));
    NC = numel(unique(D.spikeIds));


    mv = cellfun(@(x) x{1}.maxV, D.units(hasrf));
    iix = mv > 10;

    hasrf = hasrf(iix);
    nrf = numel(hasrf);

    fprintf('%s has %d/%d with RFs\n', subj, nrf, NC)

    for i = 1:nrf
        rf = D.units{hasrf(i)}{1};
        if strcmp(subj, 'allen')
            rf.contour(:,1) = rf.contour(:,1) - 50;
        end
        rf.contour = rf.contour + randn*.2;
        plot(rf.contour(:,1), rf.contour(:,2), 'Color', [cmap(6,:) .5]); hold on
    end

    xlim([-50 50])
    ylim([-40 40])
    grid on

    plot(0, 0, '+k')
    plot.formatFig(gcf, [1.25 1], 'nature')
    set(gca, 'XTick', -50:10:50)
    set(gca, 'YTick', -40:10:40)
    saveas(gcf, fullfile(figdir, sprintf('rfs_%s.pdf', subj)))
end



