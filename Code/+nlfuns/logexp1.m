function [f,df,ddf] = logexp1(x)
%  [f,df,ddf] = logexp1(x);
%
%  Computes the function:
%     f(x) = log(1+exp(x))
%  and returns first and second derivatives

f = log(1+exp(x));

if nargout > 1
    df = exp(x)./(1+exp(x));
end
if nargout > 2
    ddf = exp(x)./(1+exp(x)).^2;
end

% Check for small values
if any(x(:)<-20)
    iix = (x(:)<-20);
    f(iix) = exp(x(iix));
    df(iix) = f(iix);
    ddf(iix) = f(iix);
end

% Check for large values
if any(x(:)>20)
    iix = (x(:)>20);
    f(iix) = x(iix);
    df(iix) = 1;
    ddf(iix) = 0;
end