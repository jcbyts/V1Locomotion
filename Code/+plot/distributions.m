function dist = distributions(th, cho, varargin)
% dist = plotChoiceDistribution(th, cho, varargin)

ip = inputParser();
ip.addParameter('bins', 0:359)
ip.addParameter('densityFun', @ksdensity); %, @(x) isa(x, 'funcion_handle'))
ip.addParameter('color', repmat(.1, 1, 3))
ip.addParameter('binAcrossNDirections', 0)
ip.addParameter('FaceAlpha', .5)
ip.KeepUnmatched = true;
ip.parse(varargin{:})

ix = ~(isnan(th) | isnan(cho));
th = th(ix);
cho = cho(ix);

ths = unique(th);
n   = numel(ths);

dist = nan(numel(ip.Results.bins), n);
for i = 1:n
    
    % grab from neighboring bins -- this should not be hard
    % coded
    bins = -ip.Results.binAcrossNDirections:ip.Results.binAcrossNDirections;
    inds = mod(i + bins, n);
    inds(inds==0) = n;
    iix = ismember(th,ths(inds));
    
    tmp = cho(iix);
    
    cnt = ip.Results.densityFun(tmp(:), ip.Results.bins);
    cnt = cnt/sum(cnt);
    
    dist(:,i) = cnt;
    plot.patchFill(ip.Results.bins, cnt*30*(mean(diff(ths))), ip.Results.color, ths(i), 0, 1, 'FaceAlpha', ip.Results.FaceAlpha); hold on

end