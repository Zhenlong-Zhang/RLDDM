classdef RL_RandomRT
    properties
        Choice          
        Reward          
        ReactionTime    
        Results         
        OptimizedParams 
    end

    methods
        function obj = RL_RandomRT(choice, reward, reaction_time)
            obj.Choice = choice;
            obj.Reward = reward;
            obj.ReactionTime = reaction_time;
        end

        function obj = fit(obj)
            choice = obj.Choice;
            reward = obj.Reward;
            reaction_time = obj.ReactionTime;

            % 修正的参数初始化，确保在边界范围内
            alpha_init = 0.5;  
            beta_init = 2.5;   
            init_params = [alpha_init, beta_init];

            % 设定优化边界
            lb = [0.01, 0.1];  
            ub = [0.9, 10];    

            % 运行优化
            options = optimset('Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);
            [opt_params, total_nll] = fmincon(@(p) obj.computeNLL(p, choice, reward, reaction_time), ...
                                              init_params, [], [], [], [], lb, ub, [], options);

            % 保存优化结果
            obj.OptimizedParams = opt_params;
            obj.Results = table(opt_params(1), opt_params(2), total_nll, ...
                'VariableNames', {'Alpha', 'Beta', 'TotalNLL'});
        end

        function nll = computeNLL(obj, params, choice, reward, reaction_time)
            % 提取优化参数
            alpha = params(1);  
            beta = params(2);   

            % 初始化 Q 值
            Q = [0.5, 0.5];
            log_lik = 0;

            for t = 1:length(choice)
                % RL 选择概率
                p_1 = 1 / (1 + exp(-beta * (Q(1) - Q(2))));
                p_1 = max(min(p_1, 1 - 1e-6), 1e-6);  

                % 选择的负对数似然
                if choice(t) == 1
                    choice_prob = p_1;
                    Q(1) = Q(1) + alpha * (reward(t) - Q(1));
                else
                    choice_prob = 1 - p_1;
                    Q(2) = Q(2) + alpha * (reward(t) - Q(2));
                end

                log_lik = log_lik + log(choice_prob);
            end

            % 随机反应时间参数（固定）
            mu_random = 1;  
            lambda_random = 5;
            rt_nll = -sum(log(obj.inverseGaussianPDF(reaction_time, mu_random, lambda_random)));

            % 总 NLL
            nll = -(log_lik - rt_nll);
        end
    end

    methods (Static)
        function pdf = inverseGaussianPDF(t, mu, lambda)
            % 反高斯分布的概率密度函数
            pdf = (lambda ./ (2 * pi * t.^3)).^0.5 .* exp(-lambda .* (t - mu).^2 ./ (2 * mu^2 .* t));
            pdf(t <= 0) = 1e-50; % 避免非法值
        end
    end
end
