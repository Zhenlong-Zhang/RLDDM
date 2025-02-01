## RLDDM but onlt fit ddm part, rl is using random instead


classdef ClassMixedA < handle
    properties
        Choice       % 真实数据中的选择 (1或2)
        Reward       % 真实数据中的奖励 (用于RLDDM更新Q)
        RT           % 真实数据中的反应时 (单位: s)
        Results      % 拟合结果 (表格)
        InitialParams% 随机初始化 (或手动给定)
    end
    
    methods
        function obj = ClassMixedA(choice, reward, rt)
            % 构造函数
            if any(isnan(choice)) || any(isnan(reward)) || any(isnan(rt))
                error('Input data contains NaN values.');
            end
            obj.Choice = choice(:);
            obj.Reward = reward(:);
            obj.RT     = rt(:);
        end
        
        function obj = fit(obj)
            % 随机初始化参数
            rng('shuffle');
            init_p1   = rand() * 0.8 + 0.1;   % p1 ∈ [0.1, 0.9]
            init_alpha= rand() * 0.5 + 0.1;   % alpha ∈ [0.1, 0.6]
            init_k    = rand() * 2.0 + 0.5;   % k ∈ [0.5, 2.5]
            init_a    = rand() * 1.0 + 0.5;   % a ∈ [0.5, 1.5]
            init_t0   = rand() * 0.2 + 0.1;   % t0 ∈ [0.1, 0.3]
            
            obj.InitialParams = [init_p1, init_alpha, init_k, init_a, init_t0];
            
            % 优化设置
            lb = [0.01, 0.01, 0.1, 0.1, 0.05];
            ub = [0.99, 1.0,  3.0,  3.0, 0.5];
            options = optimset('Display', 'iter', 'TolFun', 1e-6, 'TolX', 1e-6);

            % 用 fmincon 最小化 NLL
            [best_params, best_nll] = fmincon(@(p) obj.computeNLL(p), ...
                obj.InitialParams, [], [], [], [], lb, ub, [], options);

            % 记录结果
            obj.Results = table( ...
                best_params(1), best_params(2), best_params(3), ...
                best_params(4), best_params(5), best_nll, ...
                'VariableNames', {'p1', 'alpha', 'k', 'a', 't0', 'TotalNLL'} ...
            );
        end
        
        function nll = computeNLL(obj, params)
            % 计算选择和反应时间的总 NLL
            p1    = params(1);  % Baseline: choice的概率
            alpha = params(2);  
            k     = params(3);
            a     = params(4);
            t0    = params(5);
            
            choice_data = obj.Choice;
            rt_data     = obj.RT;
            reward_data = obj.Reward;
            
            % 1) Choice NLL (Baseline)
            eps0 = 1e-50;
            logLik_choice = 0;
            for t = 1:length(choice_data)
                if choice_data(t) == 1
                    logLik_choice = logLik_choice + log(max(p1, eps0));
                else
                    logLik_choice = logLik_choice + log(max(1 - p1, eps0));
                end
            end
            choiceNLL = -logLik_choice;
            
            % 2) RT NLL (RLDDM)
            Q = [0.5, 0.5];  % 初始Q值
            logLik_rt = 0;
            for t = 1:length(rt_data)
                c = choice_data(t);  % 实际选择
                r = reward_data(t);
                rt = rt_data(t);
                
                % 漂移率计算
                v = k * (Q(1) - Q(2));
                v = max(min(v, 10), -10); % 限制漂移率范围
                
                mu_ig = max(a / abs(v), 1e-6);  % 防止数值不稳定
                lambda_ig = max(a^2, 1e-6);

                % 调整反应时间
                rt_adj = max(rt - t0, 1e-3);
                pdf_val = obj.inverseGaussianPDF(rt_adj, mu_ig, lambda_ig);
                logLik_rt = logLik_rt + log(max(pdf_val, eps0));
                
                % 更新 Q 值
                if c == 1
                    Q(1) = Q(1) + alpha * (r - Q(1));
                else
                    Q(2) = Q(2) + alpha * (r - Q(2));
                end
            end
            rtNLL = -logLik_rt;
            
            % 3) 总NLL
            nll = choiceNLL + rtNLL;
        end
        
        function pdf = inverseGaussianPDF(~, t, mu, lambda)
            % 反高斯分布 PDF
            if t <= 0 || mu <= 0 || lambda <= 0
                pdf = 1e-50;
            else
                pdf = sqrt(lambda / (2 * pi * t^3)) * ...
                      exp(-lambda * (t - mu)^2 / (2 * mu^2 * t));
            end
        end
    end
end
