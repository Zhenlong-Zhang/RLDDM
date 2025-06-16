% this is for taking out the last bin of all session before fitting

load('EphysMaleFemaleAirData.mat');
allRats = fieldnames(res.concat_data);  
folderOut = 'rtcutoff';
if ~exist(folderOut, 'dir')
    mkdir(folderOut);
end

%  Percentage bin (20 bins, trail-level)
nbins = 20;
bin_rt_values_pct = containers.Map();

for i = 1:numel(allRats)
    rat = allRats{i};
    for sec = 1:numel(res.concat_data.(rat))
        dataMat = res.concat_data.(rat){sec};
        if isempty(dataMat), continue; end
        if istable(dataMat)
            initRT = dataMat{:,5};
        else
            initRT = dataMat(:,5);
        end
        initRT = initRT(~isnan(initRT) & initRT > 0);
        n = numel(initRT);
        if n < nbins, continue; end

        edges = round(linspace(1, n+1, nbins+1));
        for b = 1:nbins
            idx_range = edges(b):(edges(b+1)-1);
            key = sprintf('Bin%02d', b);
            if ~isKey(bin_rt_values_pct, key), bin_rt_values_pct(key) = []; end
            bin_rt_values_pct(key) = [bin_rt_values_pct(key); initRT(idx_range)];
        end
    end
end

means_pct = zeros(nbins,1); sems_pct = zeros(nbins,1); delta_pct = strings(nbins,1);
for b = 1:nbins
    key = sprintf('Bin%02d', b);
    vals = bin_rt_values_pct(key);
    means_pct(b) = mean(vals);
    sems_pct(b) = std(vals) / sqrt(length(vals));
    if b > 1
        [~, p] = ttest2(vals, bin_rt_values_pct(sprintf('Bin%02d',b-1)));
        diffval = means_pct(b) - means_pct(b-1);
        star = ""; if p < 0.05, star = "*"; end
        delta_pct(b) = sprintf('+%.2f%s', diffval, star);
    else
        delta_pct(b) = " ";
    end
end

figure('Visible','off');
bar(means_pct, 'FaceColor', [0.1 0.5 0.5]); hold on;
errorbar(1:nbins, means_pct, sems_pct, 'k', 'LineStyle','none', 'LineWidth',1.5);
set(gca, 'XTick', 1:nbins, 'XTickLabel', arrayfun(@(x)sprintf('Bin%02d',x), 1:nbins, 'UniformOutput',false), 'XTickLabelRotation', 45);
ylabel('Init Latency (s)');
title('Mean Init Latency by Percentage Bin (Trail-level, No RT Drop)');
ylim([0, max(means_pct + sems_pct) + 1]);
grid on;
for i = 1:nbins
    text(i, means_pct(i) + sems_pct(i) + 0.2, delta_pct(i), ...
        'HorizontalAlignment','center', 'FontSize', 10);
end
exportgraphics(gcf, fullfile(folderOut, 'Bar_RT_by_PercentBin_Traillevel.png'), 'Resolution', 400);
