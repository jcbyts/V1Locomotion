function [L, dL] = mse_loss_mod(prs,X,Y,nlfun,g,o)
% Mean Squared Error Loss function
% [L, dL] = mse_loss(prs,X,Y,nlfun)

if numel(prs) > size(X,2)
    fitmod = true;
    modulator = prs(end-1);
    offset = prs(end);
    prs = prs(1:end-2);
    
else
    fitmod = false;
    modulator = 1;
    offset = 0;
end

xproj = X*prs;
% 
if nargin < 5
    g = 1;
end

%     offset = 0;
% end
% 
% if nargin < 5
%     modulator = 1;
% end

if nargin < 4
    yhat = xproj.*modulator + offset;
    dNL = 1*ones(size(xproj,1),1);
else
    if nargout > 1
        [yhat, dNL] = nlfun(xproj.*modulator.*g);
        yhat = yhat + offset + o;
    else
        yhat = nlfun(xproj.*modulator) + offset + o;
    end
end

nt = numel(Y);
err = Y - yhat;
L = (err'*err)/nt;

if nargout > 1
    if fitmod
        dL = -2*err'*([dNL.*modulator.*X dNL.*xproj.*g dNL])/nt;
    else
        dL = -2*err'*(dNL.*modulator.*X)/nt;
    end
end

