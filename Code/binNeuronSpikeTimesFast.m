function [Y, binfun] = binNeuronSpikeTimesFast(sp, eventTimes, binSize, keep_sparse)
% BIN SPIKE TIMES THE FASTEST WAY POSSIBLE IN MATLAB
% 
% Inputs:
%   sp [struct]: Kilosort output struct
%   has fields:
%       st [T x 1]: spike times
%       clu [T x 1]: unit id
% Outpus:
%   Y [nBins x nNeurons]
%
% Example Call:
%   Y = binNeuronSpikeTimesFast(Exp.osp, eventTimes, binsize)

if nargin < 4
    keep_sparse = false;
end

% conversion from time to bins
binfun = [];

[~, ~, ind1] = histcounts(sp.st, eventTimes);
[~, ~, ind2] = histcounts(sp.st, eventTimes+binSize);

ix = ind2+1 == ind1;
Y = sparse(ind1(ix), sp.clu(ix), ones(sum(ix), 1), numel(eventTimes), max(sp.clu));

if ~keep_sparse
    Y = full(Y);
end