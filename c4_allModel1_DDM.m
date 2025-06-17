% DDM the very basic baseline. 
% only has k for drift rate, a for decision boun 

function rlddm_fit_fixed_alpha_by_section_fminunc
    clear; clc;
    dataFolder = 'data_cleaned';
    files = dir(fullfile(dataFolder, '*.mat'));

    numRandomTries = 5;

    fixed_alpha_dict = struct( ...
        'rat136', 0.4501, 'rat137', 0.5796, 'rat139', 0.3898, ...
        'rat147', 0.5157, 'rat150', 0.4328, 'rat151', 0.5356, ...
        'rat152', 0.6396, 'rat195', 0.4667, 'rat198', 0.4985, ...
        'rat199', 0.5582, 'rat200', 0.6279, 'rat224', 0.4998, ...
        'rat225', 0.5975, 'rat228', 0.4744);

    all_results = {};  % cell array to collect all section results

    for i = 1:length(files)
        load(fullfile(dataFolder, files(i).name));
        ratID = files(i).name(1:end-4);
        data = combinedData;

        if isfield(fixed_alpha_dict, ratID)
            alpha_fixed = fixed_alpha_dict.(ratID);
        else
            error('ratID %s not found in fixed_alpha_dict.', ratID);
        end

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
            bestParams = NaN(1, 4);
            for r = 1:numRandomTries
                initParams = randn(1,4);  % unbounded space
                options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                    'MaxFunctionEvaluations',1e4,'MaxIterations',1e3);
                [optParams, fval] = fminunc(@(x) computeNLL_fixedAlpha_transformed(alpha_fixed, x, data_section), ...
                    initParams, options);
                if fval < bestNLL
                    bestNLL = fval;
                    bestParams = optParams;
                end
            end

            realParams = [exp(bestParams(1)^2), ...  % k
                          exp(bestParams(2)), ...    % a
                          1/(1+exp(-bestParams(3))), ...  % T
                          bestParams(4)];  % w: no transform, allow positive and negative

            all_results(end+1, :) = {ratID, sectionIDs(s), n_trials, alpha_fixed, ...
                realParams(1), realParams(2), realParams(3), realParams(4), bestNLL};

            fprintf('Rat: %s | Section: %d | Trials: %d | alpha=%.4f | Params=[k=%.4f, a=%.4f, T=%.4f, w=%.4f] | NLL=%.2f\n',...
                ratID, sectionIDs(s), n_trials, alpha_fixed, realParams(1), realParams(2), realParams(3), realParams(4), bestNLL);
        end
    end

    header = {'ratID', 'sectionID', 'trial_count', 'alpha', 'k', 'a', 'T', 'w', 'NLL'};
    T = cell2table(all_results, 'VariableNames', header);
    writetable(T, 'rlddm_section_results_fminunc.csv');
end

function [nll, nll_rt, nll_choice] = computeNLL_fixedAlpha_transformed(alpha, x, data)
    k = x(1);       
    a = x(2);
    T = 1 / (1 + exp(-x(3)));
    w = x(4);  % allow positive and negative

    rts     = data(:,3);
    choices = data(:,1);
    rewards = data(:,2);
    newCell = data(:,4);
    n_trials = length(rts);

    Q = [0.5, 0.5];
    logp_rt = 0;

    for t = 1:n_trials
        if newCell(t) == 1
            Q = [0.5, 0.5];
        end

        Qdiff = Q(2) - Q(1);
        v = k ;
        

        Qsum = Q(1) + Q(2);
        a_t = max(a , 0.01);

        rtObs = rts(t);
        rt_decision = rtObs - T;
        if rt_decision <= 0
            p_rt = eps;
        else
            p_rt = wfpt(rt_decision, v, a_t);
        end
        p_rt = max(p_rt, 1e-12);
        logp_rt = logp_rt + log(p_rt);

        chosenIdx = choices(t);
        R = rewards(t);
        Q(chosenIdx) = Q(chosenIdx) + alpha * (R - Q(chosenIdx));
    end

    nll_rt = -logp_rt;
    nll_choice = NaN;
    nll = nll_rt;
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
