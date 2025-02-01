classdef RandomChoice_DDM
    properties
        Choice         
        ReactionTime   
        Results        
        OptimizedParams 
    end

    methods
        function obj = RandomChoice_DDM(choice, reaction_time)
            obj.Choice = choice;
            obj.ReactionTime = max(reaction_time, 1e-3);  % 确保无负值
        end

        function obj = fit(obj)
            choice = obj.Choice;
            reaction_time = obj.ReactionTime;

            % 计算选择概率
            n_trials = length(choice);
            n_1 = sum(choice == 1);
            p_1 = n_1 / n_trials; % 选择 1 的概率

            % 选择的负对数似然
            choice_nll = -sum(log(p_1) .* (choice == 1) + log(1 - p_1) .* (choice == 2));

            % 设置 DDM 参数优化
            init_params = [1.0, 1.5, 0.2];  % 修正初始值
            lb = [0.1, 0.5, 0.1];  
            ub = [2.0, 2.0, 0.5];  

            options = optimset('Display', 'iter', 'Diagnostics', 'on', 'TolFun', 1e-6, 'TolX', 1e-6);
            [opt_params, rt_nll] = fmincon(@(p) obj.computeRTNLL(p, reaction_time), ...
                                          init_params, [], [], [], [], lb, ub, [], options);

            % 总 NLL
            total_nll = choice_nll + rt_nll;

            % 保存结果
            obj.OptimizedParams = opt_params;
            obj.Results = table(p_1, opt_params(1), opt_params(2), opt_params(3), total_nll, ...
                'VariableNames', {'P_1', 'DriftRate', 'Boundary', 'T0', 'TotalNLL'});
        end

        function nll = computeRTNLL(obj, params, reaction_time)
            try
                drift_rate = params(1);
                boundary = params(2);
                t0 = params(3);

                rt_adj = max(reaction_time - t0, 1e-3); % 确保反应时间非零

                mu = max(abs(boundary / abs(drift_rate)), 1e-6);
                lambda = max(boundary^2, 1e-6);
                rt_pdf = RandomChoice_DDM.inverseGaussianPDF(rt_adj, mu, lambda);

                nll = -sum(log(rt_pdf));

            catch ME
                disp('Error in computeRTNLL:');
                disp(ME.message);
                nll = Inf;  % 避免优化器失败
            end
        end
    end

    methods (Static)
        function pdf = inverseGaussianPDF(t, mu, lambda)
            pdf = (lambda ./ (2 * pi * t.^3)).^0.5 .* exp(-lambda .* (t - mu).^2 ./ (2 * mu^2 .* t));
            pdf(t <= 0) = 1e-50; % 避免非法值
        end
    end
end
