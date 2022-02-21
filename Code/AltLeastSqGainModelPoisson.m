function [Betas, Gain, Ridge, Rhat, Lgain, Lfull, g] = AltLeastSqGainModelPoisson(X, Y, train_inds, covIdx, covLabels, StimLabel, GainLabel, restLabels, Lgain, nlfun)
% Use alternating least squares to estimate gains and offsets
% [Betas, Gain, Ridge] = AltLeastSqGainModel(X, Y, covIdx, covLabels, StimLabel, GainLabel)

% StimLabel = 'Stim';
% GainLabel = 'Is Running';
if ~exist('nlfun', 'var')
    nlfun = @nlfuns.expfun;
end

gain_idx = ismember(covIdx, find(ismember(covLabels, GainLabel)));
stim_idx = ismember(covIdx, find(ismember(covLabels, StimLabel)));
if exist('restLabels', 'var')
    rest_idx = ismember(covIdx, find(ismember(covLabels, restLabels)));
else
    rest_idx = ~(gain_idx | stim_idx); % everything besides stim and gain
end

g0 = 0; % initialize gain

if exist('restLabels', 'var') && ismember(GainLabel, restLabels)
    gdim = sum(gain_idx);
    w0 = ones(gdim,1);
    gain_has_weights = true;
    gain_rest_idx = find(gain_idx(rest_idx));
else
    gain_has_weights = false;
end

if gain_has_weights
    xgain = X(train_inds,gain_idx)*w0;
else
    xgain = X(train_inds,gain_idx);
end

xstim = X(train_inds,stim_idx);
xrest = X(train_inds,rest_idx);
Ytrain = Y(train_inds);
% SSfull0 = sum( (Ytrain - mean(Ytrain)).^2);
g = 1 + xgain*g0;

if ~exist('Lgain', 'var')
    Lgain = nan;
end

if ~exist('Lfull', 'var')
    Lfull = nan;
end

ndstim = sum(stim_idx);
ndrest = sum(rest_idx);
ridgeii = 2:(ndstim+ndrest+1);

[Bfull,Lfull] = regression.autoRegress_PoissonRidge([ones(numel(train_inds),1) xstim.*g xrest],Ytrain, nlfun, ridgeii,.1,[.1 1 10]);

% [Lfull, Bfull] = ridgeMML(Ytrain, [xstim.*g xrest], false, Lfull);

yhatF = nlfun([xstim.*g xrest]*Bfull(2:end) + Bfull(1));
bps = bitsPerSpike(yhatF, Ytrain, mean(Ytrain));

fminopts = optimset()

bps0 = bps;
steptol = 1e-3;
iter = 1;

while true
    
    % fit gains
    stimproj = xstim*Bfull((1:sum(stim_idx))+1);
    if gain_has_weights
        w0 = Bfull(1 + sum(stim_idx) + gain_rest_idx);
        xgain = X(train_inds,gain_idx)*w0;
    end
    
    op
    [Lgain, Bgain] = ridgeMML(Ytrain, [stimproj stimproj.*xgain xrest], false, Lgain);

    g0 = Bgain(2:3);
    g = g0(1) + xgain*g0(2);
    
    [Lfull, Bfull0] = ridgeMML(Ytrain, [xstim.*g xrest], false, Lfull);
    
    yhatF = [xstim.*g xrest]*Bfull(2:end) + Bfull(1);
    SSfull = sum( (Ytrain - yhatF).^2);

    step = SSfull0 - SSfull;
    SSfull0 = SSfull;
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

Rhat = [xstim.*g xrest]*Betas(2:end) + Betas(1);

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