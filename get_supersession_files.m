%% import super sessions


fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
fdir = fullfile(fpath, 'gratings');

fprintf('')

subjs = {'brie', 'gru'};
for isubj = 1:numel(subjs)
    subj = subjs{isubj};
    import_supersession(subj, fdir)
    
    % move file over
    fname = sprintf('%sD_all.mat', subj);
    movefile(fullfile(fdir, fname), fullfile(fpath, fname))
end

%% Import for the Allen Institute data
% If the data is not already in the path, run 
fdir = fullfile(fpath, 'brain_observatory_1.1');
subj = 'allen';
import_supersession(subj, fdir)

% move file over
fname = sprintf('%sD_all.mat', subj);
movefile(fullfile(fdir, fname), fullfile(fpath, fname))