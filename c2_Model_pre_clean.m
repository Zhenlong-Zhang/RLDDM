% this is for taking out the last bin of all session before fitting
% To run the model on different latency types, please update dataMat{:,x}. see below
% Column 5(dataMat{:,5})corresponds to initial latency; column 6 (dataMat{:,6})corresponds to choice latency.


clear; clc;
filePath = 'EphysMaleFemaleAirData.mat';
dataStruct = load(filePath);
if ~isfield(dataStruct, 'res')
    error('No "res" variable found in the MAT file.');
end
res = dataStruct.res;
if ~isfield(res, 'concat_data')
    error('res does not contain "concat_data".');
end

ratFields = fieldnames(res.concat_data);
outputFolder = 'data_cleaned';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:numel(ratFields)
    ratName = ratFields{i};
    ratCell = res.concat_data.(ratName);
    combinedData = [];

    for j = 1:numel(ratCell)
        dataMat = ratCell{j};
        if istable(dataMat)
            choice = dataMat{:,1};
            reward = dataMat{:,2};
            % change to rt = dataMat{:,6};to see for choice latency. 5 is c init latency and 6 is choice
            rt = dataMat{:,5};
        else
            choice = dataMat(:,1);
            reward = dataMat(:,2);
            % change to rt = dataMat(:,6); to see for choice latency. 
            rt = dataMat(:,5);
        end

        % --- RT lower bound only ---
        rt(rt < 0.01) = 0.01;

        % --- Remove last 5% of trials ---
        N = length(rt);
        cutoff = floor(0.95 * N);
        choice = choice(1:cutoff);
        reward = reward(1:cutoff);
        rt = rt(1:cutoff);

        % --- mark session start ---
        newCellIndicator = zeros(size(rt));
        if ~isempty(rt)
            newCellIndicator(1) = 1;
        end

        cellData = [choice, reward, rt, newCellIndicator];
        combinedData = [combinedData; cellData];
    end

    save(fullfile(outputFolder, [ratName, '.mat']), 'combinedData');
end
