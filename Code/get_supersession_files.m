%% import super sessions


fpath = getpref('FREEVIEWING', 'HUKLAB_DATASHARE');
fdir = fullfile(fpath, 'gratings');

fprintf('Building supersession files. This can be slow...\n')

subjs = {'brie', 'gru', 'allen'};
for isubj = 1:numel(subjs)

    subj = subjs{isubj};
    import_supersession(subj, fdir)
    
    % move file over
    fname = sprintf('%sD_all.mat', subj);
    movefile(fullfile(fdir, fname), fullfile(fpath, fname))
end

fprintf('\n\nDone with marmosets\n\n')

%% Import for the Allen Institute data
% If the data is not already in the grating path, run allen_data_to_matlab.py
% fprintf('Building supersession struct for the Allen Institute data\n')
% fdir = fullfile(fpath, 'brain_observatory_1.1');
% subj = 'allen';
% import_supersession(subj, fdir)
% 
% % move file over
% fname = sprintf('%sD_all.mat', subj);
% movefile(fullfile(fdir, fname), fullfile(fpath, fname))