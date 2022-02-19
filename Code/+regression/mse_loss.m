function [L, dL] = mse_loss(prs,X,Y,nlfun)
% Mean Squared Error Loss function
% [L, dL] = mse_loss(prs,X,Y,nlfun)

xproj = X*prs;

if nargin < 4
    yhat = xproj;
    dNL = 1;
else
    if nargout > 1
        [yhat, dNL] = nlfun(xproj);
    else
        yhat = nlfun(xproj);
    end
end

nt = numel(Y);
err = Y - yhat;
L = (err'*err)/nt;

if nargout > 1
    dL = -2*err'*(dNL.*X)/nt;
end

