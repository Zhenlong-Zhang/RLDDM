## very basic rlddm model rl part has only learning rate, ddm part only has decision boundry, k for the difference in q value, and t0 for non decision time. 
## p(chioce) is 1 / (1 + exp(-1 * v * a)), and rt is using decision boundry and drift rate(found by k) to build a distrution in inverseGaussian

classdef ClassBasicRLDDM
    properties
        Choice          
        Reward          
        ReactionTime    
        InitialParams   
        Results         
    end

    methods
        function obj = ClassBasicRLDDM(choice, reward, reaction_time)
            % 构造函数
            if any(isnan(choice)) || any(isnan(reward)) || any(isnan(reaction_time))
                error('Input data contains NaN values.');
            end
            if any(reaction_time <= 0)
                error('Reaction time contains non-positive values.');
            end

            obj.Choice = choice;
            obj.Reward = reward;
            obj.ReactionTime = reaction_time;
        end

        function obj = fit(obj)
            choice = obj.Choice;
            reward = obj.Reward;
            reaction_time = obj.ReactionTime;

            % 随机初始化参数：alpha, k, a, t0
            rng('shuffle'); 
            init_alpha = rand() * 0.89 + 0.01; % alpha ∈ [0.01, 0.9]
            init_k = rand() * 2 + 0.5;         % k ∈ [0.5, 2.5]
            init_a = rand() * 1.9 + 0.1;       % a ∈ [0.1, 2.0]
            init_t0 = rand() * 0.2 + 0.1;      % t0 ∈ [0.1, 0.3]

            obj.InitialParams = [init_alpha, init_k, init_a, init_t0];

            % 检查初始点的 NLL
            disp('Fitting RLDDM Model...');
            disp(['Initial Parameters: Alpha=', num2str(init_alpha), ...
                  ', k=', num2str(init_k), ', a=', num2str(init_a), ...
                  ', t0=', num2str(init_t0)]);
            try
                init_nll = obj.computeNLL(obj.InitialParams, choice, reward, reaction_time);
                disp(['Initial NLL: ', num2str(init_nll)]);
            catch ME
                disp('Error evaluating initial NLL:');
                rethrow(ME);
            end

            % 优化参数
            lb = [0.01, 0.5, 0.1, 0.1];  % 下界
            ub = [0.9,  2.5, 2.0, 0.3];  % 上界
            options = optimset('Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);
            [params_hat, total_nll] = fmincon(@(p) obj.computeNLL(p, choice, reward, reaction_time), ...
                                              obj.InitialParams, [], [], [], [], ...
                                              lb, ub, [], options);

            % 保存结果
            obj.Results = table(params_hat(1), ...
                                params_hat(2), ...
                                params_hat(3), ...
                                params_hat(4), ...
                                ...
                                total_nll, ...
                                'VariableNames', {'learning rate','k','decision boundry','t0','TotalNLL'});
        end

        function obj = fitMultiple(obj, num_repeats)
            best_nll = inf; 
            best_params = []; 

            for i = 1:num_repeats
                disp(['Fitting iteration ', num2str(i), ' of ', num2str(num_repeats), '...']);
                try
                    temp_obj = obj.fit();
                catch ME
                    disp(['Error in iteration ', num2str(i)]);
                    continue;
                end

                % 验证参数范围
                if temp_obj.Results.TotalNLL < best_nll
                    best_nll = temp_obj.Results.TotalNLL;
                    best_params = temp_obj.Results; 
                end
            end

            if isempty(best_params)
                error('All iterations failed.');
            end

            obj.Results = best_params;
        end

        function nll = computeNLL(obj, params, choice, reward, reaction_time)
            % 计算 RLDDM 的 NLL
            alpha = params(1);  
            k = params(2);      
            a = params(3);      
            t0 = params(4);     

            Q = [0.5, 0.5];  % 初始 Q 值
            log_lik = 0;

            for t = 1:length(choice)
                try
                    % 漂移率计算
                    v = k * (Q(1) - Q(2));
                    v = max(min(v, 10), -10); % 防止 v 数值过大或过小

                    % 选择概率
                    p_1 = 1 / (1 + exp(-1 * v * a));
                    p_1 = max(min(p_1, 1 - 1e-6), 1e-6); % 确保概率范围合法

                    if choice(t) == 1
                        choice_prob = p_1;
                    else
                        choice_prob = 1 - p_1;
                    end

                    % 修正反应时
                    rt_adj = max(reaction_time(t) - t0, 1e-3); % 确保反应时间非负
                    mu = max(abs(a / abs(v)), 1e-6);
                    lambda = max(a^2, 1e-6);
                    rt_pdf = obj.inverseGaussianPDF(rt_adj, mu, lambda);

                    % 累加对数似然
                    log_lik = log_lik + log(choice_prob) + log(rt_pdf);

                    % Q 值更新
                    if choice(t) == 1
                        Q(1) = Q(1) + alpha * (reward(t) - Q(1));
                    else
                        Q(2) = Q(2) + alpha * (reward(t) - Q(2));
                    end
                catch ME
                    disp(['Error at trial ', num2str(t)]);
                    disp(['v: ', num2str(v), ', p_1: ', num2str(p_1), ...
                          ', choice_prob: ', num2str(choice_prob), ...
                          ', rt_adj: ', num2str(rt_adj), ...
                          ', mu: ', num2str(mu), ', lambda: ', num2str(lambda)]);
                    rethrow(ME);
                end
            end

            nll = -log_lik; % 返回负对数似然
        end

        function pdf = inverseGaussianPDF(~, t, mu, lambda)
            % 反高斯分布的概率密度函数
            if t <= 0 || mu <= 0 || lambda <= 0
                pdf = 1e-50; % 避免非法值
            else
                pdf = max((lambda ./ (2 * pi * t.^3)).^0.5 ...
                      .* exp(-lambda .* (t - mu).^2 ./ (2 * mu^2 .* t)), 1e-50);
            end
        end
    end
end
