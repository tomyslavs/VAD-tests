function [featuresTraining, maskTrainingCat, featuresValidation, maskValidationCat] = split_train_validation(featuresTraining, maskTrainingCat,win_len,overlap,ratio,lenValid);
% ratio - validatio ratio, 0.2 means 20% for validation 80% for training
% lenVal - length in seconds for a validation segment, 10 means 10s validation and 40s train segment
 
NfeaturesPerSecond = round(1/(win_len*(1-overlap)));
NfeaturesPerValidSegment = NfeaturesPerSecond*lenValid;
NfeaturesPerTrainSegment = round(NfeaturesPerValidSegment*(1-ratio)/ratio);
NfeaturesPerTrainValidSegment = NfeaturesPerTrainSegment + NfeaturesPerValidSegment;

fTraining = [];
mTraining = [];
fValidation = [];
mValidation = [];

sizeT = size(featuresTraining,1);
Nfull = fix(sizeT/NfeaturesPerTrainValidSegment);
Nlast = mod(sizeT,NfeaturesPerTrainValidSegment);

for i=1:Nfull
    k = (i-1)*NfeaturesPerTrainValidSegment;
    fTraining = cat(1,fTraining,featuresTraining(k+1:k+NfeaturesPerTrainSegment,:));
    mTraining = cat(1,mTraining,maskTrainingCat(k+1:k+NfeaturesPerTrainSegment));
    fValidation = cat(1,fValidation,featuresTraining(k+NfeaturesPerTrainSegment+1:k+NfeaturesPerTrainValidSegment,:));
    mValidation = cat(1,mValidation,maskTrainingCat(k+NfeaturesPerTrainSegment+1:k+NfeaturesPerTrainValidSegment));
end
if (Nlast>0) % add last features
    if (Nlast<=NfeaturesPerTrainSegment) % if Nlast<=1600, add Train only
        fTraining = cat(1,fTraining,featuresTraining(Nfull*NfeaturesPerTrainValidSegment+1:Nfull*NfeaturesPerTrainValidSegment+Nlast,:));
        mTraining = cat(1,mTraining,maskTrainingCat(Nfull*NfeaturesPerTrainValidSegment+1:Nfull*NfeaturesPerTrainValidSegment+Nlast));
    end
    if (Nlast>NfeaturesPerTrainSegment) % if Nlast>1600, add Train and Valid
        fTraining = cat(1,fTraining,featuresTraining(Nfull*NfeaturesPerTrainValidSegment+1:Nfull*NfeaturesPerTrainValidSegment+NfeaturesPerTrainSegment,:));
        mTraining = cat(1,mTraining,maskTrainingCat(Nfull*NfeaturesPerTrainValidSegment+1:Nfull*NfeaturesPerTrainValidSegment+NfeaturesPerTrainSegment));
        fValidation = cat(1,fValidation,featuresTraining(Nfull*NfeaturesPerTrainValidSegment+NfeaturesPerTrainSegment+1:sizeT,:));
        mValidation = cat(1,mValidation,maskTrainingCat(Nfull*NfeaturesPerTrainValidSegment+NfeaturesPerTrainSegment+1:sizeT));
    end
end

% for i=1:sizeT % Perrasyti koda, ilgai split'ina
%     k = mod(i,NfeaturesPerTrainValidSegment); % sitas sukasi 1:2000,
%     if (k>1 && k<=NfeaturesPerTrainSegment) % kai pataiko i 1:1600->Train
%         fTraining = cat(1,fTraining,featuresTraining(i,:)); % pridek i-tosios eilutes visus pozymiu stulpelius
%         mTraining = cat(1,mTraining,maskTrainingCat(i));
%     else % 1601:2000->Valid
%         fValidation = cat(1,fValidation,featuresTraining(i,:)); % pridek i-tosios eilutes visus pozymiu stulpelius
%         mValidation = cat(1,mValidation,maskTrainingCat(i));
%     end
% end

featuresTraining = fTraining;
maskTrainingCat = mTraining;
featuresValidation = fValidation;
maskValidationCat = mValidation;

