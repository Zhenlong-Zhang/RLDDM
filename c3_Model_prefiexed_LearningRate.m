% output learning rates for later fixed fitting
  
  function fit_simple_rl_allrats()
    clc; clear;

    
    dataFolder = 'data_cleaned';
    files = dir(fullfile(dataFolder, '*.mat'));


    lb = [0.01, 0.01];  % alpha, beta
    ub = [0.99, 10.00];
    numTries = 5;

    for i = 1:length(files)
     
        filePath = fullfile(dataFolder, files(i).name);
        load(filePath);  

        if ~exist('combinedData', 'var')
            warning('No variable named "combinedData" in %s', files(i).name);
            continue;
        end

       
        data = combinedData(:, 1:2);  % choice, reward
        ratID = files(i).name(1:end-4);

       
        bestNLL = Inf;
        bestParams = [];

        for r = 1:numTries
            init = rand(1,2).*(ub - lb) + lb;
            options = optimoptions('fmincon', 'Display','off','Algorithm','sqp');
            [params, fval] = fmincon(@(x) computeNLL(x, data), init, [], [], [], [], lb, ub, [], options);

            if fval < bestNLL
                bestNLL = fval;
                bestParams = params;
            end
        end

        
        fprintf('Rat: %s | alpha = %.4f, beta = %.4f | NLL = %.2f\n', ...
            ratID, bestParams(1), bestParams(2), bestNLL);
    end
end

function nll = computeNLL(params, data)
    alpha = params(1);
    beta = params(2);
    Q = [0.5, 0.5];
    nll = 0;

    for t = 1:size(data,1)
        choice = data(t,1);
        reward = data(t,2);

        p = softmax(Q, beta);
        prob = max(p(choice), 1e-5); 
        nll = nll - log(prob);

      
        Q(choice) = Q(choice) + alpha * (reward - Q(choice));
    end
end

function p = softmax(Q, beta)
    expQ = exp(beta .* Q);
    p = expQ ./ sum(expQ);
end
