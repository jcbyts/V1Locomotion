%% import super sessions


fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
fdir = fullfile(fpath, 'gratings');

subjs = {'brie', 'gru'};
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    import_supersession(subj, fdir)
    
    % move file over
    fname = sprintf('%sD_all.mat', subj);
    movefile(fullfile(fdir, fname), fullfile(fpath, fname))
end
%%

fdir = fullfile(fpath, 'brain_observatory_1.1');
subj = 'allen';
import_supersession(subj, fdir)

% move file over
fname = sprintf('%sD_all.mat', subj);
movefile(fullfile(fdir, fname), fullfile(fpath, fname))