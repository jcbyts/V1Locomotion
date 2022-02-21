function [L,dL] = neglogpost_GLM(wts,hprs,mstruct)
% [L,dL,ddL] = neglogpost_GLM(wts,hprs,mstruct)
%
% Compute negative log-posterior of data under GLM regression model,
% plus gradient and Hessian
%
% INPUTS:
%   wts [m x 1] - regression weights
%  hprs [p x 1] - hyper-parameters for prior
%       mstruct - model structure with fields
%        .neglogli - func handle for negative log-likelihood
%        .logprior - func handle for log-prior 
%        .liargs - cell array with fixed args to neg log-likelihood
%        .priargs - cell array with fixed args to log-prior function
%
% OUTPUTS:
%    L [1 x 1] - negative log-posterior
%   dL [m x 1] - Gradient


if nargout <= 1
    L = mstruct.neglogli(wts,mstruct.liargs{:});
    L = L-mstruct.logprior(wts,hprs,mstruct.priargs{:});

elseif nargout == 2
    [L,dL] = mstruct.neglogli(wts,mstruct.liargs{:});
    [p,dp] = mstruct.logprior(wts,hprs,mstruct.priargs{:});
    L = L(:)-p(:);
    dL = dL(:)-dp(:);

end