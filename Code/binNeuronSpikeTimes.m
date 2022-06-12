function Y = binNeuronSpikeTimes(sp, eventTimes, binSize)
% BIN SPIKE TIMES THE SLOW WAY
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


nEvents = numel(eventTimes);
cids = unique(sp.clu);
NC = numel(cids);

Y = nan(nEvents, NC);
for i = 1:nEvents
    ix = sp.st > eventTimes(i) & sp.st < (eventTimes(i) + binSize);

    Y(i,:) = sum(sp.clu(ix) == cids);
end