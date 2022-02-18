function [pval, stats] = boot_ttest(x,y,metric, nboot, seed, two_sided)
% two-sided bootstrapped ttest

if nargin < 6
    two_sided = true;
end

if nargin < 5
    seed = 1234;
end

rng(seed)

if nargin < 4
    nboot = 500;
end

if ~exist('metric', 'var') || isempty(metric)
    metric = @mean;
end

x = x(~isnan(x));
y = y(~isnan(y));

nx = numel(x);
ny = numel(y);
nt = nx + ny;

nullgroup = [x(:); y(:)];


null = metric( nullgroup(randi(nt, [nx nboot])) ) - metric( nullgroup(randi(nt, [ny nboot])) );
mdiff = metric(x) - metric(y);

propgt = mean(mdiff > null);
pval = propgt;
if two_sided
    pval(pval>.5) = 1 - pval(pval>.5);
    pval = pval*2;
end

stats = struct('nx', nx, 'ny', ny, 'nboot', nboot, 'mudiff', mdiff, 'propgt', propgt);