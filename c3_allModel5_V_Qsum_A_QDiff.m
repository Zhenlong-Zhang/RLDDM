% drift = k*qsum*sign(qdiff)
% decision bound = a - qdiff


function rlddm_fit_fixed_params_by_section_fminunc
    clear; clc;
    dataFolder = 'data_cleaned';
    files = dir(fullfile(dataFolder, '*.mat'));
 
    numRandomTries = 5;
 
    fixed_params = struct( ...
        'rat136', [0.479178, 0.358262, 0.732817, 0.998366], ...
        'rat137', [0.284785, 0.660138, 0.497956, 0.677551], ...
        'rat139', [0.376044, 0.333612, 0.644675, 0.997644], ...
        'rat147', [0.295795, 0.348021, 0.864621, 0.993438], ...
        'rat150', [0.26766,  0.491997, 0.460677, 0.574123], ...
        'rat151', [0.468853, 0.411894, 0.598276, 0.998598], ...
        'rat152', [0.237988, 0.176460, 0.587010, 0.915566], ...
        'rat195', [0.343575, 0.435140, 0.615049, 0.923953], ...
        'rat198', [0.399536, 0.558971, 0.557204, 0.990524], ...
        'rat199', [0.559672, 0.480746, 0.642962, 0.998023], ...
        'rat200', [0.496227, 0.612836, 0.651010, 0.993467], ...
        'rat224', [0.331848, 0.448323, 0.795225, 0.869242], ...
        'rat225', [0.470567, 0.383536, 0.790753, 0.995127], ...
        'rat228', [0.294514, 0.490056, 0.586377, 0.886918] ...
    );
 
    all_results = {};  % collect results
 
    for i = 1:length(files)
        load(fullfile(dataFolder, files(i).name));
        ratID = files(i).name(1:end-4);
        data = combinedData;
 
        if isfield(fixed_params, ratID)
            q_params = fixed_params.(ratID);
        else
            error('ratID %s not found in fixed_params.', ratID);
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
                initParams = randn(1,4);
                options = optimoptions('fminunc','Display','off','Algorithm','quasi-newton', ...
                    'MaxFunctionEvaluations',1e4,'MaxIterations',1e3);
                [optParams, fval] = fminunc(@(x) computeNLL_fixedParams(x, data_section, q_params), ...
                    initParams, options);
                if fval < bestNLL
                    bestNLL = fval;
                    bestParams = optParams;
                end
            end
            safeLog = @(x) log(max(abs(x), eps));  % eps 2.2e-16
 
            realParams = [ ...
                10 / (1 + exp(-safeLog(bestParams(1)))), ...  % k  (0,10)
                10 / (1 + exp(-safeLog(bestParams(2)))), ...  % a  (0,10)
                1  / (1 + exp(-safeLog(bestParams(3)))), ...  % T  (0,1)
                bestParams(4)                               % w 
            ];
 
 
            all_results(end+1, :) = {ratID, sectionIDs(s), n_trials, ...
                q_params(1), q_params(2), q_params(3), q_params(4), ...
                realParams(1), realParams(2), realParams(3), realParams(4), bestNLL};
 
            fprintf('Rat: %s | Section: %d | Trials: %d | LR+=%.4f, LR-=%.4f | Params=[k=%.4f, a=%.4f, T=%.4f, w=%.4f] | NLL=%.2f\n',...
                ratID, sectionIDs(s), n_trials, q_params(1), q_params(2), realParams(1), realParams(2), realParams(3), realParams(4), bestNLL);
        end
    end
 
    header = {'ratID', 'sectionID', 'trial_count', ...
              'alpha_pos', 'alpha_neg', 'decay_pos', 'decay_neg', ...
              'k', 'a', 'T', 'w', 'NLL'};
    T = cell2table(all_results, 'VariableNames', header);
    writetable(T, 'model5.csv');
end
 

 % computing loglikehood
function [nll, nll_rt, nll_choice] = computeNLL_fixedParams(x, data, q_params)
    k = x(1);
    a = x(2);
    T = x(3);
    w = x(4);
 
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
        Qsum = Q(1) + Q(2);
        Qdiff = Q(2) - Q(1);
        v = k * sign(Qdiff) * Qsum;
        if choices(t) == 1
            v = -v;
        end
 
        a_t = max(a - w * Qdiff, 0.01);
 
        rtObs = rts(t);
        rt_decision = rtObs - T;
        if rt_decision <= 0
            p_rt = eps;
        else
            p_rt = wfpt(rt_decision, v, a_t);
        end
        p_rt = max(p_rt, 1e-12);
        logp_rt = logp_rt + log(p_rt);
 
        Q = update_q(Q, choices(t), rewards(t), q_params);
    end
 
    nll_rt = -logp_rt;
    nll_choice = NaN;
    nll = nll_rt;
end

% q updating
function q = update_q(q, c, r, params)
    alpha_pos = params(1);
    alpha_neg = params(2);
    decay_pos = params(3);
    decay_neg = params(4);
 
    if r > 0
        q(c) = q(c) + alpha_pos * (r - q(c));
        q(3 - c) = (1 - decay_pos) * q(3 - c);
    else
        q(c) = q(c) + alpha_neg * (r - q(c));
        q(3 - c) = (1 - decay_neg) * q(3 - c);
    end
end
 


% First passage time for Wiener diffusion model. Approximation based on
% Navarro & Fuss (2009).
function P = wfpt(t, v, a, w, err)
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

