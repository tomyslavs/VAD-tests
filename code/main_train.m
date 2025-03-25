% clc; clear all; close all;
 
%% Options, parameters
win_len = 0.05; % apdorojamo slenkancio lango dydis, s
overlap = 0.5;  % 0.5 = 50%

%% Get features and category vectors
[afe, featuresTraining, maskTrainingCat, yvf] = give_me_a_features2(win_len,overlap,'t'); % t-train | v-validation
[afe, featuresValidation, maskValidationCat, yvf] = give_me_a_features2(win_len,overlap,'v'); % t-train | v-validation

%% Split features and labels into training cells
sequenceLength = 400;
sequenceOverlap = round(0.75*sequenceLength);

trainFeatureCell = helperFeatureVector2Sequence(featuresTraining',sequenceLength,sequenceOverlap);
trainLabelCell = helperFeatureVector2Sequence(maskTrainingCat',sequenceLength,sequenceOverlap);

%% Define LSTM
layers = [...
    sequenceInputLayer(size(featuresValidation,2))
    bilstmLayer(400,"OutputMode","sequence") % 200
    bilstmLayer(400,"OutputMode","sequence") % 200
%     bilstmLayer(400,"OutputMode","sequence") % 200
    fullyConnectedLayer(800)
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];

maxEpochs = 40; % 20
miniBatchSize = 50; % 64-Out of memory on GPU device. 32 for 9f-200-200 | 20 for 9f-400-400 | 16 for 13f-400-400 | 64 for 13f-100-100 | 128 for 13f-50-50
% checkpointPath = './nets';
options = trainingOptions("adam", ...
    "ExecutionEnvironment","auto", ... % "cpu" | 'auto' (the default value) To train on a GPU, if available.
    "MaxEpochs",maxEpochs, ...
    "MiniBatchSize",miniBatchSize, ...
    "Shuffle","every-epoch", ...
    "Verbose",0, ...
    "SequenceLength",sequenceLength, ...
    "ValidationFrequency",200,...%floor(numel(trainFeatureCell)/miniBatchSize), ...
    "ValidationData",{featuresValidation.',maskValidationCat.'}, ...
    "Plots","training-progress", ...
    "LearnRateSchedule","piecewise");
%     "CheckpointPath",checkpointPath, ...
%     "LearnRateDropFactor",0.99, ... % 0.1
%     "LearnRateDropPeriod",5

%% Train
[speechDetectNet,netInfo] = trainNetwork(trainFeatureCell,trainLabelCell,layers,options);
fprintf("Validation accuracy: %f percent.\n", netInfo.FinalValidationAccuracy);

