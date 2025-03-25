% load C:\vad2024\train_01.mat % jei reikia i workspace ikrauti issaugotus .mat tinkla ir kintamuosius 
 
%% Classify
validRange = 1:47020;%1:size(featuresValidation,1);% 300000:310000;
EstimatedVADMask = classify(speechDetectNet,featuresValidation(validRange,:).');
EstimatedVADMask = double(EstimatedVADMask);
EstimatedVADMask = EstimatedVADMask.' - 1;

%% Decision, write time to file
Dtime = make_decision(EstimatedVADMask',win_len,1);
writematrix(Dtime,'laikai.txt','Delimiter','tab');

%% Plot results
figure(10);
cm = confusionchart(double(maskValidationCat(validRange))-1,EstimatedVADMask,"title","Validation Accuracy");
cm.ColumnSummary = "column-normalized";
cm.RowSummary = "row-normalized";

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

