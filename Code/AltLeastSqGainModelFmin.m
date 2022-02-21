function [Betas, Gain, Ridge, Rhat, Lgain, Lfull, g] = AltLeastSqGainModelFmin(X, Y, train_inds, covIdx, covLabels, StimLabel, GainLabel, restLabels, Lgain, Lfull, nlfun)
% Use alternating least squares to estimate gains and offsets
% [Betas, Gain, Ridge] = AltLeastSqGainModel(X, Y, covIdx, covLabels, StimLabel, GainLabel)

% StimLabel = 'Stim';
% GainLabel = 'Is Running';
if ~exist('nlfun', 'var')
    nlfun = @nlfuns.linear;
end

gain_idx = ismember(covIdx, find(ismember(covLabels, GainLabel)));
stim_idx = ismember(covIdx, find(ismember(covLabels, StimLabel)));
if exist('restLabels', 'var')
    rest_idx = ismember(covIdx, find(ismember(covLabels, restLabels)));
else
    rest_idx = ~(gain_idx | stim_idx); % everything besides stim and gain
end

g0 = 0; % initialize gain

if sum(gain_idx) > 1 %exist('restLabels', 'var') && ismember(GainLabel, restLabels)
    gdim = sum(gain_idx);
    w0 = ones(gdim,1);
    gain_has_weights = true;
    gain_rest_idx = 1:gdim;
else
    gain_has_weights = false;
end

if gain_has_weights
    xgainraw = X(train_inds,gain_idx);
    xgain = xgainraw*w0;
else
    xgain = X(train_inds,gain_idx);
end

xstim = X(train_inds,stim_idx);
xrest = X(train_inds,rest_idx);
Ytrain = Y(train_inds);
% SSfull0 = sum( (Ytrain - mean(Ytrain)).^2);
g = 1 + xgain*g0;

if ~exist('Lgain', 'var') || isempty(Lgain)
    Lgain = .01;
end

if ~exist('Lfull', 'var') || isempty(Lfull)
    Lfull = .01;
end


biascol = ones(numel(train_inds),1);
fminopts = optimset('MaxIter', 5e3, 'GradObj', 'On', 'Hessian', 'lbfgs', 'display', 'off');
%       mstruct - model structure with fields
%        .neglogli - func handle for negative log-likelihood
%        .logprior - func handle for log-prior 
%        .liargs - cell array with fixed args to neg log-likelihood
%        .priargs - cell array with fixed args to log-prior function
if isempty(xgain)
    
    xtrain = [biascol xstim xrest];
    n = size(xtrain,2);
    Cinv = diag(ones(n,1)); Cinv(1) = .001;

    Betas0 = (xtrain'*xtrain + Cinv)\(xtrain'*Ytrain);
%     [Lfull, Betas0] = ridgeMML(Ytrain, [xstim xrest], false, Lfull);
    Ridge = Lfull;
    

    mstruct = struct();
    mstruct.neglogli = @regression.mse_loss;
    mstruct.logprior = @regression.logprior_ridge;
    mstruct.liargs = {xtrain, Ytrain, nlfun};
    mstruct.priargs ={2:numel(Betas0),.1};

    fun = @(w) regression.neglogpost_GLM(w, Lfull, mstruct);
    Betas = fmincon(fun, Betas0, [],[],[],[],[],[],[],fminopts);
    
    xstim = X(:,stim_idx);
    xrest = X(:,rest_idx);
%     Rhat0 = [xstim xrest]*Betas0(2:end) + Betas0(1);
    Rhat = nlfun([xstim xrest]*Betas(2:end) + Betas(1));
    Gain = [nan nan];
    Lgain = nan;
    g = nan(size(Rhat));
    return
else
    g = 1 + xgain*g0;
    g = max(g, 0);
end

xtrain = [biascol xstim.*g xrest];

n = size(xtrain,2);
Cinv1 = diag(ones(n,1)); Cinv1(1) = .001;
Betas0 = (xtrain'*xtrain + Cinv1)\(xtrain'*Ytrain);
mstruct = struct();
mstruct.neglogli = @regression.mse_loss;
mstruct.logprior = @regression.logprior_ridge;
mstruct.liargs = {xtrain, Ytrain, nlfun};
mstruct.priargs ={2:numel(Betas0),.1};

fun = @(w) regression.neglogpost_GLM(w, Lfull, mstruct);
Bfull = fmincon(fun, Betas0, [],[],[],[],[],[],[],fminopts);

yhatF = nlfun([xstim.*g xrest]*Bfull(2:end) + Bfull(1));
MSEfull = mean( (Ytrain - yhatF).^2);

MSEfull0 = MSEfull;
steptol = 1e-3;
iter = 1;

while true
    
    % fit gains
    stimproj = xstim*Bfull((1:sum(stim_idx))+1);

    if gain_has_weights
        xtmp = [biascol stimproj stimproj .* xgainraw xrest];
        n = size(xtmp,2);
        Cinv2 = diag(ones(n,1)); Cinv2(1) = 0.001;
        Betas0 = (xtmp'*xtmp + Cinv2)\(xtmp'*Ytrain);

        mstruct = struct();
        mstruct.neglogli = @regression.mse_loss;
        mstruct.logprior = @regression.logprior_ridge;
        mstruct.liargs = {xtmp, Ytrain, nlfun};
        mstruct.priargs ={2:numel(Betas0),.1};
        lb = [-1e3 -1e3 zeros(1, size(xgainraw,2)) -1e3*ones(1, size(xgainraw,2))];
        fun = @(w) regression.neglogpost_GLM(w, Lgain, mstruct);
        Bgain = fmincon(fun, Betas0, [],[],[],[],lb,[],[],fminopts);
%         [Lgain, Bgain] = ridgeMML(Ytrain, xtmp, false, Lgain);

        w0 = Bgain(2+gain_rest_idx);
        xgain = 1 + X(train_inds,gain_idx)*w0;
%         xgain = xgain ./ max(xgain) * 3;
%         figure(1);
%         plot(xgain); hold on
        g0 = [Bgain(2) 1];
        g = xgain;
    else
        xtmp = [biascol stimproj stimproj.*xgain xrest];

        n = size(xtmp,2);
        Cinv2 = diag(ones(n,1)); Cinv2(1) = 0.001;
        Betas0 = (xtmp'*xtmp + Cinv2)\(xtmp'*Ytrain);

        mstruct = struct();
        mstruct.neglogli = @regression.mse_loss;
        mstruct.logprior = @regression.logprior_ridge;
        mstruct.liargs = {xtmp, Ytrain, nlfun};
        mstruct.priargs ={2:numel(Betas0),.1};

        fun = @(w) regression.neglogpost_GLM(w, Lgain, mstruct);
        Bgain = fmincon(fun, Betas0, [],[],[],[],[],[],[],fminopts);

%         [Lgain, Bgain] = ridgeMML(Ytrain, [stimproj stimproj.*xgain xrest], false, Lgain);

        g0 = Bgain(2:3);
        g0 = max(g0, 0);
        g0 = min(g0, 5);

        g = g0(1) + xgain*g0(2);
    end
    g = max(g, 0); % gain cannot go negative





    %%
    xtrain = [biascol xstim.*g xrest];

    n = size(xtrain,2);
    Cinv1 = diag(ones(n,1)); Cinv1(1) = .001;
    Betas0 = (xtrain'*xtrain + Cinv1)\(xtrain'*Ytrain);
    mstruct = struct();
    mstruct.neglogli = @regression.mse_loss;
    mstruct.logprior = @regression.logprior_ridge;
    mstruct.liargs = {xtrain, Ytrain, nlfun};
    mstruct.priargs ={2:numel(Betas0),.1};

    fun = @(w) regression.neglogpost_GLM(w, .01, mstruct);
    Bfull0 = fmincon(fun, Betas0, [],[],[],[],[],[],[],fminopts);

    yhatF = nlfun([xstim.*g xrest]*Bfull0(2:end) + Bfull0(1));
    MSEfull = mean( (Ytrain - yhatF).^2);
    %%
%     [Lfull, Bfull0] = ridgeMML(Ytrain, [xstim.*g xrest], false, Lfull);
%     xxg = [ones(nt,1) xstim.*g xrest];
%     fun = @(w) regression.mse_loss(w, xxg, Ytrain, nlfun);
%     Bfull0 = fmincon(fun, Bfull0, [], [],[],[],[],[],[],fminopts);
% 
%     MSEfull = fun(Bfull0);
    
    step = MSEfull0 - MSEfull;
    MSEfull0 = MSEfull;
    if step < steptol || iter > 5
        break
    end
    fprintf("Step %d, %02.5f\n", iter, step)
    iter = iter + 1;
    Bfull = Bfull0;
    g1 = g0;
    if gain_has_weights
        w1 = w0;
    end
end

if ~exist('g1', 'var')
    Bfull = Bfull0;
    g1 = g0;
    if gain_has_weights
        w1 = w0;
    end
end

Gain = g1;
Betas = Bfull;
Ridge = Lfull;

if gain_has_weights
    xgain = X(:,gain_idx)*w1;
else
    xgain = X(:,gain_idx);
end
xstim = X(:,stim_idx);
xrest = X(:,rest_idx);
g = Gain(1) + xgain*Gain(2);

Rhat = nlfun([xstim.*g xrest]*Betas(2:end) + Betas(1));

% 
% %% different parameterization
% cc = 2;
% 
% g0 = 1; % initialize gain
% 
% xgain = X(:,gain_idx);
% xstim = X(:,stim_idx);
% xrest = X(:,~(gain_idx | stim_idx));
% SSfull0 = sum( (Y - mean(Y)).^2);
% 
% g = xgain*g0 + ~xgain;
% 
% Lgain = nan;
% Lfull = nan;
% 
% [Lfull, Bfull, convergenceFailures] = ridgeMML(Y, [xstim.*g xrest], false, Lfull);
% 
% yhatF = [xstim.*g xrest]*Bfull(2:end) + Bfull(1);
% SSfull = sum( (Y - yhatF).^2);
% 
% step = SSfull0 - SSfull;
% SSfull0 = SSfull;
% steptol = 1e-3;
% iter = 1;
% 
% %%
% 
% while step > steptol && iter < 10
%     
%     % fit gains
%     stimproj = xstim*Bfull(find(stim_idx)+1);
% 
%     [Lgain, Bgain] = ridgeMML(Y(xgain>0), [stimproj(xgain>0) xrest(xgain>0,:)], false, Lgain);
% 
%     g0 = Bgain(2);
%     g = xgain*g0 + ~xgain;
%     
%     [Lfull, Bfull] = ridgeMML(Y, [xstim.*g xrest], false);
%     
%     yhatF = [xstim.*g xrest]*Bfull(2:end) + Bfull(1);
%     SSfull = sum( (Y - yhatF).^2);
% 
%     step = SSfull0 - SSfull;
%     SSfull0 = SSfull;
%     fprintf("Step %d, %02.5f\n", iter, step)
%     iter = iter + 1;
% end
% 
%%