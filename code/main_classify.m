% clc; clear all; close all;
% Uzkauna anksciau apmokyta tinkla su 2024-11-30 datos irasais (visa workspace'a) ir testuoja
% naujus 05_10 irasus (po viena irasa is eiles), isveda grafikus
 
%% Load workspace with trained net
% load C:\vad2024\train_01.mat

%% Get features and category vectors
% \05_10\ 20 irasu   \05_19\ 12 irasu   \05_24\ 15 irasu   \06_02\ 22 irasu   \06_10\ 24 irasu
% give_me_a_features3.m f-joje keisti irasu/labelu kataloga (wav ir txt viename folderyje 'C:\Users\T\Downloads\05_10\')
NRec = 100; % 18 vnt = 20% nuo visu irasu
for j=1:NRec
    [afe, featuresValidation, maskValidationCat, yvf, lname] = give_me_a_features3(win_len,overlap,'v',j); % t-train | v-validation

    %% Classify
    Range = 26000;
    if (size(featuresValidation,1)<Range) % jei langu maziau uz 26000
        validRange = 1:size(featuresValidation,1); % tai paimk is karto visa clasifikavimui
    else
        validRange = 1:Range; % tai paimk pirma range'a
    end
    vr = ceil(size(featuresValidation,1)/max(validRange)); % kiek batchu po #Range langu, nes gpu nepaveza ilgo iraso suklasifikuoti.
    EstimatedVADMask = [];
    for i=1:vr % per visus batchus
        EstimatedVADMask_r = classify(speechDetectNet,featuresValidation(validRange,:).');
        EstimatedVADMask = cat(2,EstimatedVADMask,EstimatedVADMask_r);
        r_start = max(validRange)+1;
        r_end = r_start+Range-1;
        if (r_end > size(featuresValidation,1))
            r_end = size(featuresValidation,1);
        end
        validRange = r_start:r_end;
    end
    validRange = 1:size(featuresValidation,1);
    EstimatedVADMask = double(EstimatedVADMask);
    EstimatedVADMask = EstimatedVADMask.' - 1;

    %% Decision, write time to file
    Dtime = make_decision(EstimatedVADMask',win_len*overlap,0.1);
    new_name = cat(2,cat(2,lname,'_classified'),'.txt');
    writematrix(Dtime,new_name,'Delimiter','tab');

    %% Plot results
    figure(10);
    cm = confusionchart(double(maskValidationCat(validRange))-1,EstimatedVADMask,"title","Validation Accuracy");
    cm.ColumnSummary = "column-normalized";
    cm.RowSummary = "row-normalized";

    %% Save confusionchart
    f = gcf;
    fname = cat(2,num2str(j),'_confusion');
    fname = cat(2,fname,'.png');
    exportgraphics(f,fname,'Resolution',100);

    %% Speech Detection
    decisionsWindow = 1.2*EstimatedVADMask.';
    decisionsSample = [repelem(decisionsWindow(1),numel(afe.Window)), ...
                       repelem(decisionsWindow(2:end),numel(afe.Window)-afe.OverlapLength)];
    decisionsWindowGT = 1.1*(double(maskValidationCat(validRange).')-1);
    decisionsSampleGT = [repelem(decisionsWindowGT(1),numel(afe.Window)), ...
                       repelem(decisionsWindowGT(2:end),numel(afe.Window)-afe.OverlapLength)];

    figure(11);
    t = (0:numel(decisionsSample)-1)/afe.SampleRate;
    plot(...t,yvf(1:numel(t)),
         t,decisionsSample);
    hold on;
    plot(t,decisionsSampleGT(1:numel(t)),'LineWidth',2);
    hold off;
    xlabel('Laikas, s');
    ylabel('Amplitude');
    legend('LSTM out','Ground truth');%'Filtruotas',
    
    %% Save decisionchart
    f = gcf;
    fname = cat(2,num2str(j),'_decision');
    fname = cat(2,fname,'.png');
    exportgraphics(f,fname,'Resolution',100);
end