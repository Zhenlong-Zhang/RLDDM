classdef RL_DDM
    properties
        Choice          
        Reward          
        ReactionTime    
        Results         
        OptimizedParams 
    end

    methods
        function obj = RL_DDM(choice, reward, reaction_time)
            obj.Choice = choice;
            obj.Reward = reward;
            obj.ReactionTime = reaction_time;
        end

        function obj = fit(obj)
            choice = obj.Choice;
            reward = obj.Reward;
            reaction_time = obj.ReactionTime;

            % 参数初始化
            alpha_init = 0.5;  
            beta_init = 2.5;   
            drift_init = 1.0;
            boundary_init = 1.5;
            t0_init = 0.3;
            init_params = [alpha_init, beta_init, drift_init, boundary_init, t0_init]; 

            % 设定优化边界
            lb = [0.01, 0.1, 0.1, 0.5, 0.1];  
            ub = [0.9, 10, 2.0, 2.0, 0.5];    

            % 运行优化
            options = optimset('Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);
            [opt_params, total_nll] = fmincon(@(p) obj.computeNLL(p, choice, reward, reaction_time), ...
                                              init_params, [], [], [], [], lb, ub, [], options);

            % 保存优化结果
            obj.OptimizedParams = opt_params;
            obj.Results = table(opt_params(1), opt_params(2), opt_params(3), opt_params(4), opt_params(5), total_nll, ...
                'VariableNames', {'Alpha', 'Beta', 'DriftRate', 'Boundary', 'T0', 'TotalNLL'});
        end

        function nll = computeNLL(~, params, choice, reward, reaction_time)
            alpha = params(1);  
            beta = params(2);   
            drift_rate = params(3);
            boundary = params(4);
            t0 = params(5);

            % --------- 1. RL 模型 NLL 计算 ---------
            Q = [0.5, 0.5];  % 初始化 Q 值
            choice_nll = 0;
            eps0 = 1e-6;  % 避免 log(0) 错误

            for t = 1:length(choice)
                p_1 = 1 / (1 + exp(-beta * (Q(1) - Q(2))));
                p_1 = max(min(p_1, 1 - eps0), eps0); % 确保概率范围 [eps0, 1-eps0]

                if choice(t) == 1
                    choice_prob = p_1;
                    Q(1) = Q(1) + alpha * (reward(t) - Q(1));
                else
                    choice_prob = 1 - p_1;
                    Q(2) = Q(2) + alpha * (reward(t) - Q(2));
                end

                choice_nll = choice_nll - log(choice_prob);
            end

            % --------- 2. DDM 模型 NLL 计算 ---------
            rt_adj = max(reaction_time - t0, eps0);  % 修正反应时间，避免负值
            mu = max(abs(boundary / abs(drift_rate)), eps0);  % 计算 mu
            lambda = max(boundary^2, eps0);  % 计算 lambda

            rt_pdf = RL_DDM.inverseGaussianPDF(rt_adj, mu, lambda);
            rt_nll = -sum(log(rt_pdf + eps0));  % 计算 RT 的负对数似然

            % --------- 3. 总 NLL 计算 ---------
            nll = choice_nll + rt_nll;  % 将 RL 和 DDM 的 NLL 相加
        end
    end

    methods (Static)
        function pdf = inverseGaussianPDF(t, mu, lambda)
            % 反高斯分布的概率密度函数
            eps0 = 1e-50;  % 避免非法值
            pdf = (lambda ./ (2 * pi * t.^3)).^0.5 .* exp(-lambda .* (t - mu).^2 ./ (2 * mu^2 .* t));
            pdf(t <= 0) = eps0; % 确保 RT > 0
        end
    end
end