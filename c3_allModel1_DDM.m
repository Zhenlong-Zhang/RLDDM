% DDM the very basic baseline. 
% only has k for drift rate, a for decision boun T for non decision time. 
function ddm_fit_by_section_fminunc
    clear; clc;
    dataFolder = 'data_cleaned';
    files = dir(fullfile(dataFolder, '*.mat'));
 
    numRandomTries = 5;
 
    all_results = {};  % collect results
 
    for i = 1:length(files)
        load(fullfile(dataFolder, files(i).name));
        ratID = files(i).name(1:end-4);
        data = combinedData;
 
        section_id = cumsum(data(:,4) == 1);
        data = [data, section_id];
 
        sectionIDs = unique(section_id);
        for s = 1:length(sectionIDs)
            sectionIdx = data(:,5) == sectionIDs(s);
            data_section = data(sectionIdx, :);
            n_trials = sum(sectionIdx);
 
            if n_trials < 20
                continue;
            end
 
            bestNLL = Inf;
            bestParams = NaN(1, 3);
            for r = 1:numRandomTries
                initParams = randn(1,3);  % k, a, T
                options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                    'MaxFunctionEvaluations',1e4,'MaxIterations',1e3);
                [optParams, fval] = fminunc(@(x) computeNLL_ddm(x, data_section), ...
                    initParams, options);
                if fval < bestNLL
                    bestNLL = fval;
                    bestParams = optParams;
                end
            end
 
            realParams = [exp(bestParams(1)^2), ...  % k
                          exp(bestParams(2)), ...    % a
                          1/(1+exp(-bestParams(3)))];  % T
 
            all_results(end+1, :) = {ratID, sectionIDs(s), n_trials, ...
                realParams(1), realParams(2), realParams(3), bestNLL};
 
            fprintf('Rat: %s | Section: %d | Trials: %d | Params=[k=%.4f, a=%.4f, T=%.4f] | NLL=%.2f\n',...
                ratID, sectionIDs(s), n_trials, realParams(1), realParams(2), realParams(3), bestNLL);
        end
    end
 
    header = {'ratID', 'sectionID', 'trial_count', 'k', 'a', 'T', 'NLL'};
    T = cell2table(all_results, 'VariableNames', header);
    writetable(T, 'model1.csv');
end
 
function nll = computeNLL_ddm(x, data)
    k = x(1);
    a = x(2);
    T = 1 / (1 + exp(-x(3)));  % constrain T > 0
 
    rts = data(:,3);
    n_trials = length(rts);
 
    logp_rt = 0;
 
    for t = 1:n_trials
        v = k;  % constant drift rate for DDM
        a_t = max(a, 0.01);  % prevent zero boundary
 
        rtObs = rts(t);
        rt_decision = rtObs - T;
 
        if rt_decision <= 0
            p_rt = eps;
        else
            p_rt = wfpt(rt_decision, v, a_t);
        end
        p_rt = max(p_rt, 1e-12);
        logp_rt = logp_rt + log(p_rt);
    end
 
    nll = -logp_rt;
end
 
% First passage time for Wiener diffusion model. Approximation based on
% Navarro & Fuss (2009).
% USAGE: p = wfpt(t,v,a,z,err)
    %
    % INPUTS:
    %   t - hitting time (e.g., response time in seconds)
    %   v - drift rate
    %   a - threshold
    %   w - bias (default: 0.5)
    %   err - error threshold (default: 1e-6)
    %
    % OUTPUTS:
    %   p - probability density
function P = wfpt(t, v, a, w, err)
    % Default arguments
    if nargin < 4; w = 0.5; end
    if nargin < 5; err = 1e-6; end
 
    P = zeros(size(t));
    for i = 1:length(t)
        tt = t(i) / (a^2);
        if pi * tt * err < 1
            kl = sqrt(-2 * log(pi * tt * err) / (pi^2 * tt));
            kl = max(kl, 1 / (pi * sqrt(tt)));
        else
            kl = 1 / (pi * sqrt(tt));
        end
 
        if 2 * sqrt(2 * pi * tt) * err < 1
            ks = 2 + sqrt(-2 * tt * log(2 * sqrt(2 * pi * tt) * err));
            ks = max(ks, sqrt(tt) + 1);
        else
            ks = 2;
        end
 
        p = 0;
        if ks < kl
            K = ceil(ks);
            for k = -floor((K - 1) / 2):ceil((K - 1) / 2)
                p = p + (w + 2 * k) * exp(-((w + 2 * k)^2) / (2 * tt));
            end
            p = p / sqrt(2 * pi * tt^3);
        else
            K = ceil(kl);
            for k = 1:K
                p = p + k * exp(-(k^2) * (pi^2) * tt / 2) * sin(k * pi * w);
            end
            p = p * pi;
        end
 
        P(i) = p * exp(-v * a * w - (v^2) * t(i) / 2) / (a^2);
    end
end

