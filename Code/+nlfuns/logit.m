function x = logit(p)
% x = logit(p)
%
% Compute logit function:
%   x = log(p./(1-p));
%
% Input:  p \in (0,1)
% Output: x \in Realsx  = log(p./(1-p));

x = log(p./(1-p));