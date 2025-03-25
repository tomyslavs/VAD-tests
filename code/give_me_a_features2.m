function [afe, features, mask, yf] = give_me_a_features2(win_len,overlap,type)

%% Read list of records
% home = 'C:\Users\T\Downloads\sounds20240728\'; % home:koefi or work:T % Senas
% home = 'C:\Users\T\Downloads\sounds20240830\'; % home:koefi or work:T % Naujas
% home = 'C:\Users\T\Downloads\05_10\'; % home:koefi or work:T % \05_10\ turi 20 irasu
% home = 'C:\Users\T\Downloads\05_19\'; % home:koefi or work:T % \05_19\ turi 12 irasu
% home = 'C:\Users\T\Downloads\05_24\'; % home:koefi or work:T % \05_24\ turi 15 irasu
home = 'C:\vad2024\records\';
if strcmp('t',type)
    name_t = cat(2,home,'records_for_training.txt');
elseif strcmp('v',type)
    name_t = cat(2,home,'records_for_validation.txt');
else
    disp('Wrong type of features: "t" or "v"');
end 
name_t
fileID = fopen(name_t,'r');
wav_names = textscan(fileID,'%s','Delimiter','\n');
fclose(fileID);
viso_train_irasu = numel(wav_names{1,1});

%% Initialization
featuresTrainingM = [];
maskTrainingCatM = [];

%% Loop that code through all wav files and get training set
for i=1:viso_train_irasu
    tic;
    %% Read audio
    name = wav_names{1}{i};
    %name_r = cat(2,home,name); % dabar full path yra 'records_for_training.txt'
    name_r = cat(2,name,'.wav'); % file name
    [y,fs] = audioread(name_r);
    y = y/max(abs(y));
    
%     if strcmp('v',type) % kai validacijai naudojame 2021-04-30 irasus
%         name_l = cat(2,home,name); 
        name_l = cat(2,name,'.txt'); % label name
%     else
%         name_l = cat(2,home,'audacity\'); name_l = cat(2,name_l,name); name_l = cat(2,name_l,'_data\');
%         name_l = cat(2,name_l,'Label Track.txt');
%     end
    labels = readmatrix(name_l);
    
    disp([num2str(i) '/' num2str(viso_train_irasu) ' ' num2str(length(y)/fs/60) ' min, ' name]);

    %% Fill label vector: 0-no speech, 1-speech, according Audacity labels [start end]...
    maskTraining = zeros(length(y),1);
%     maskValidation = zeros(length(yv),1);
    for j=1:size(labels,1) % scan over all rows
        if labels(j,1) < length(y)/fs %
            for k=fix(labels(j,1)*fs):fix(labels(j,2)*fs) % labeled speech
                maskTraining(k,1)=1; % fill with ones
            end
        else
            break;
        end
    end
%     maskValidation = maskTraining(length(yt)+1:end,1); %[10-12]min
%     maskTraining = maskTraining(1:length(yt),1); %[0-10]min

    %% Filter
    fc1 = 80;
    fc2 = 450;
    [b,a] = cheby1(3,3,[fc1 fc2]/(fs/2),'bandpass');
    yf = filter(b,a,y);
%     yvf = filter(b,a,yv);

    %% Feature Extractor iki 2024-12-01
%     afe = audioFeatureExtractor('SampleRate',fs, ...
%         'Window',hann(fs*win_len,"Periodic"), ...
%         'OverlapLength',overlap*fs*win_len, ...
%         ...
%         'spectralCentroid',true, ...
%         'spectralCrest',true, ...
%         'spectralEntropy',true, ...
%         'spectralFlux',true, ...
%         'spectralKurtosis',true, ...
%         'spectralRolloffPoint',true, ...
%         'spectralSkewness',true, ...
%         'spectralSlope',true, ...
%         'harmonicRatio',true);
    afe = audioFeatureExtractor('SampleRate',fs, ...
        'Window',hann(fs*win_len,"Periodic"), ...
        'OverlapLength',overlap*fs*win_len, ...
        ...
        'mfcc',true);

    featuresTraining = extract(afe,yf);
    % [numWindows,numFeatures] = size(featuresTraining);
    M = mean(featuresTraining,1);
    S = std(featuresTraining,[],1);
    featuresTraining = (featuresTraining - M) ./ S;
% 
%     featuresValidation = extract(afe,yvf);
%     % [numWindows,numFeatures] = size(featuresValidation);
%     M = mean(featuresValidation,1);
%     S = std(featuresValidation,[],1);
%     featuresValidation = (featuresValidation - M) ./ S;
% 
    %% Form desired class output vector
    windowLength = numel(afe.Window);
    hopLength = windowLength - afe.OverlapLength;
    range = (hopLength) * (1:size(featuresTraining,1)) + hopLength;
    maskMode = zeros(size(range));
    for index = 1:numel(range)
        maskMode(index) = mode(maskTraining((index-1)*hopLength+1:(index-1)*hopLength+windowLength)); % mode - most frequent value in array
    end
    maskTraining = maskMode.';
    maskTrainingCat = categorical(maskTraining);

%     range = (hopLength) * (1:size(featuresValidation,1)) + hopLength;
%     maskMode = zeros(size(range));
%     for index = 1:numel(range)
%         maskMode(index) = mode(maskValidation( (index-1)*hopLength+1:(index-1)*hopLength+windowLength )); % mode - most frequent value in array
%     end
%     maskValidation = maskMode.';
%     maskValidationCat = categorical(maskValidation);

    featuresTrainingM = cat(1,featuresTrainingM,featuresTraining);
    maskTrainingCatM = cat(1,maskTrainingCatM,maskTrainingCat);
    toc;
end

features = featuresTrainingM;
mask = maskTrainingCatM;

end