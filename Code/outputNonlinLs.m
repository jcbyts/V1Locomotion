function [what, bctrs, f] = outputNonlinLs(x, y, opts)
% [w, b, f] = outputNonlinLs(x, y)

if ~exist('opts', 'var')
    opts = struct();
end

defopts = struct('funtol', .1, 'maxiter', 5, 'print', false, 'display', 'off', 'lambda', 0.1);
opts = mergeStruct(defopts, opts);

iix = ~isnan(y);
y = y(iix);
x = x(iix,:);
X = [ones(sum(iix), 1) x]; % augment with bias

% x = [x(iix,:) ones(sum(iix),1)];
[L, what] = ridgeMML(y, x, false);
yhat = X*what;
mse0 = mean( (y - yhat).^2);

% w2 = ridge(y, [ones(numel(y), 1) x], L, 0)
% [what, ~, mse0] = lscov(x, y);
iter = 1;


while true
    % fit nonlinearity
    yhat = X*what;
    bctrs1 = quantile(yhat, linspace(0, 1, 10));
    b = tent_basis(yhat, bctrs1);
    [~, bw] = ridgeMML(y, b, false);
%     bw = lscov(b, y);
    
    % refit weights
    f1 = @(x) tent_basis(x, bctrs1)*bw(2:end) + bw(1);
    loss = @(w) sum( (y-f1(X*w)).^2) + L*norm(w);

    [what1, ~] = fminunc(loss, what, struct('Display', opts.display, 'MaxIter', 1e3));
    
    mse1 = mean( (y - X*what1).^2 );
    
    if (mse0-mse1) < opts.funtol || iter > opts.maxiter
        break
    end
    if opts.print
        fprintf('iter %d: %02.3f\n', iter, (mse0-mse1))
    end
    mse0 = mse1;
    what = what1;
    bctrs = bctrs1;
    f = f1;

    iter = iter + 1;

end

if ~exist('bctrs', 'var')
    bctrs = bctrs1;
    f = f1;
end