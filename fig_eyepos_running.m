% NOTE: This script is under construction
fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
figdir = 'Figures/HuklabTreadmill/manuscript/';

if ~exist('output', 'dir')
    mkdir('output')
end
% where the analyese will print to
fid = fopen('output/fig_session_corr.txt', 'w');

%% Eye position
thresh = 3; % running threshold

subjects = {'gru', 'brie'};

nsubjs = numel(subjects);

isubj = 1;
% load super session
D = load_subject(subjects{isubj});
cmap = getcolormap(subjects{isubj}, false);

% % or, load single session
% flist = dir(fullfile(fpath, 'gratings', 'gru*'));
% D = load(fullfile(flist(end).folder, flist(end).name));

%% get basic saccade stats
sacstarts = filter([1;-1], 1, D.eyeLabels==2);
sacon = find(sacstarts==1);
sacoff = find(sacstarts==-1);
bad = find(any(isnan(D.eyePos(sacon,1:2)),2));
sacon(bad) = [];
sacoff(bad) = [];

eyeVel = filter([1;-1], 1, D.eyePos(:,1:2));
eyeVel = hypot(eyeVel(:,1), eyeVel(:,2));

sacdurms = sacoff - sacon;
peakvel = arrayfun(@(x,y) max(eyeVel(x:y)), sacon, sacoff)/1e-3;
dx = arrayfun(@(x,y) D.eyePos(y,1)-D.eyePos(x,1), sacon, sacoff);
dy = arrayfun(@(x,y) D.eyePos(y,2)-D.eyePos(x,2), sacon, sacoff);
[sacdir, sacamp] = cart2pol(dx, dy);

bad = sacdurms > 200 | sacamp > 20 | peakvel > 2000; % these are all unreasonable values and are artifacts of something 
sacdurms(bad) = [];
peakvel(bad) = [];
dx(bad) = [];
dy(bad) = [];
sacdir(bad) = [];
sacamp(bad) = [];
sacon(bad) = [];
sacoff(bad) = [];

figure(isubj); clf
subplot(2,2,1)
histogram(sacdurms, 'binEdges', linspace(0, 200, 100), 'Normalization', 'pdf', 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1)
xlabel('saccade duration (ms)')
ylabel('pdf')

subplot(2,2,2)
ampbins = 0:.25:20;
velbins = 0:50:2000;
C = histcounts2(sacamp, peakvel, ampbins, velbins);
C = log10(C');
C(C < 0) = 0; % correct negative infinities
h = imagesc(ampbins, velbins, C); axis xy
h.AlphaData = C ./ max(C(:)) * 1;

colormap(cmap)
xlabel('amplitude amplitude')
ylabel('peak velocity')

subplot(2,2,3)
polarhistogram(sacdir, 200, 'Normalization', 'pdf', 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
title('saccade direction', 'Fontweight', 'normal')
ax = gca;
ax.FontSize = 5;

subplot(2,2,4)
xax = -6:.1:6;
C = histcounts2(dx, dy , xax, xax);
C = log10(C');
C(C < 0) = 0; % correct negative infinities
h = imagesc(xax, xax, C); axis xy
h.AlphaData = C ./ max(C(:)) * 1;

title('saccade endpoints', 'Fontweight', 'normal')
xlabel('azimuth (d.v.a.)')
ylabel('elevation (d.v.a.)')

plot.formatFig(gcf, [3 3], 'nature')
saveas(gcf, fullfile(figdir, sprintf('saccade_summary_%s.pdf', subjects{isubj})))

%%
nboot = 10;
fid = 1;

% get treadmill speed at each saccade
iix = ~(isnan(D.treadSpeed) | isnan(D.treadTime));
treadSpeed = interp1(D.treadTime(iix), D.treadSpeed(iix), D.eyeTime, 'nearest');
speedatsac = arrayfun(@(x,y) nanmean(treadSpeed(x:y)), sacon, sacoff);


run_thresh = 3;
runix = speedatsac > run_thresh;
statix = abs(speedatsac) < run_thresh;

% find running epochs
figure(isubj); clf
subplot(2,2,1)
[frun,xi] = ksdensity(sacdurms(runix));
fstat = ksdensity(sacdurms(statix), xi);
plot(xi, frun, 'Color', cmap(6,:)); hold on
plot(xi, fstat, 'Color', cmap(end,:))

mrun = median(sacdurms(runix));
mrunci = [nan, nan];
% mrunci = bootci(nboot, @median, sacdurms(runix));
mstat = median(sacdurms(statix));
mstatci = [nan, nan];
% mstatci = bootci(nboot, @median, sacdurms(statix));

y = max(ylim)*1.1;
plot(mrun, y, 'v', 'Color', cmap(6,:), 'MarkerFaceColor', cmap(6,:), 'MarkerSize', 3);
plot(mstat, y, 'v', 'Color', cmap(end,:), 'MarkerFaceColor', 'none', 'MarkerSize', 3);


[pval, h, wstats] = ranksum(sacdurms(runix), sacdurms(statix));
if h
    fprintf(fid, 'Saccade duration medians are significantly different for running %02.3f [%02.3f, %02.3f] and stationary %02.3f [%02.3f, %02.3f] \n', mrun, mrunci(1), mrunci(2), mstat, mstatci(1), mstatci(2));
    fprintf(fid, 'p=%d, ranksum=%d\n', pval, wstats.ranksum);
else
    fprintf(fid, 'Saccade duration medians are NOT significantly different for running %02.3f [%02.3f, %02.3f] and stationary %02.3f [%02.3f, %02.3f] \n', mrun, mrunci(1), mrunci(2), mstat, mstatci(1), mstatci(2));
    fprintf(fid, 'p=%d, ranksum=%d\n', pval, wstats.ranksum);
end


xlabel('saccade duration (ms)')
ylabel('pdf')

ax = subplot(2,2,2);
ampbins = 0:.25:20;
velbins = 0:50:2000;
C = histcounts2(sacamp(runix), peakvel(runix), ampbins, velbins);
C = log10(C');
C(C < 0) = 0; % correct negative infinities
h = imagesc(ampbins, velbins, C); axis xy
h.AlphaData = sqrt(C ./ max(C(:))) * 1;
colormap(cmap(1:7,:))
xd = xlim;
yd = ylim;

axclone = axes();
axclone.Position = ax.Position;

C = histcounts2(sacamp(statix), peakvel(statix), ampbins, velbins);
C = log10(C');
C(C < 0) = 0; % correct negative infinities
[c,h] = contourf(ampbins(1:end-1), velbins(1:end-1), C); axis xy
% h.AlphaData = C ./ max(C(:)) * .5;
h.AlphaData = sqrt(C ./ max(C(:))) * .5;
colormap(axclone, plot.coolwarm);
xlim(xd)
ylim(yd)
axclone.XLim = ax.XLim;
axclone.YLim = ax.YLim;
axclone.Color = 'none';

lmrun = fitlm(sacamp(runix), peakvel(runix));
wcirun = coefCI(lmrun);
lmstat = fitlm(sacamp(statix), peakvel(statix));
wcistat = coefCI(lmstat);

xlabel('amplitude amplitude')
ylabel('peak velocity')

subplot(2,2,3)
polarhistogram(sacdir, 200, 'Normalization', 'pdf', 'FaceColor', cmap(6,:), 'EdgeColor', 'none', 'FaceAlpha', 1);
title('saccade direction', 'Fontweight', 'normal')
ax = gca;
ax.FontSize = 5;

subplot(2,2,4)
xax = -6:.1:6;
C = histcounts2(dx, dy , xax, xax);
C = log10(C');
C(C < 0) = 0; % correct negative infinities
h = imagesc(xax, xax, C); axis xy
h.AlphaData = C ./ max(C(:)) * 1;

title('saccade endpoints', 'Fontweight', 'normal')
xlabel('azimuth (d.v.a.)')
ylabel('elevation (d.v.a.)')

plot.formatFig(gcf, [3 3], 'nature')
saveas(gcf, fullfile(figdir, sprintf('saccade_summary_running_%s.pdf', subjects{isubj})))



%%

amp = hypot(dx, dy);
figure(4); clf
histogram(amp)
%%

[pval, k, K] = circ_kuipertest(thR(~isnan(thR)), thS(isnan(thS)), 1, true)

% plot(dx, dy, '.')










% plot(eyeVel(sacon(i):sacoff(i),:))