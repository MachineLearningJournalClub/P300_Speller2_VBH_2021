clear all; close all

addpath('Data');
Fs=250;
channels=[1 2 3 4 5 6 7 8];
channelNames={'Fz','C3','Cz','C4','Pz','PO7','Oz','PO8','sum'};

resamplePar=6;
startS=0.15;
endS=0.55;

trainTrialsNo=60;
balanceFlag=[0 1 2]; %0-no, 1-reduced, 2-enlarged
percentageOfFeatures=0.5;

for balanceInd=1:length(balanceFlag)
    balancingFlag=balanceFlag(balanceInd);
    for subject=1:5
        fileName=['S', num2str(subject) '.mat'];
        
        load(fileName);
        
        Ind=find(trig==1 | trig==-1);
        startSample=fix(startS*Fs);
        endSample=fix(endS*Fs);
        
        baseline=mean(y(1:Ind(1),:));
        
        additionalFeatures=[];
        targetsVector=[];
        for flash=1:length(Ind)
            signal=y(Ind(flash)+startSample:Ind(flash)+endSample-1,channels)-baseline;
            signal=resample(signal,1,resamplePar);
            targetStruct{flash}=signal;
            [value, indMax]=max(signal);
            additionalFeatures(flash,:)=[max(signal)-min(signal) indMax];
            if trig(Ind(flash))==1
                targetsVector=[targetsVector 1];
            else
                targetsVector=[targetsVector 0];
            end
        end
        
        dataMatrix=[];
        for flash=1:length(Ind)
            dataMatrix=[dataMatrix; targetStruct{flash}(:)'];
        end
        dataMatrix=[dataMatrix additionalFeatures];
        
        %Normalisation!!!
        
        indexes=crossvalind('Kfold', length(targetsVector), 5);
        indTrain=1:16*trainTrialsNo;
        indTest=16*trainTrialsNo+1:16*75;
        
        featuresTrain=dataMatrix(indTrain,:);
        classesTrain=targetsVector(indTrain);
        
        featuresTest=dataMatrix(indTest,:);
        classesTest=targetsVector(indTest);
        
        if balancingFlag==1
            %Balancing the training data set by removing 12 nonTarget rows from each trial
            k=0;
            for i=1:16:length(classesTrain)
                for j=1:16
                    if classesTrain(i+j-1)==1
                        k=k+1;
                        balancedFeaturesTrain(k,:)=featuresTrain(i+j-1,:);
                        balancedclassesTrain(k)=classesTrain(i+j-1);
                    end
                end
                noOfZeros=0;
                for j=1:16
                    if classesTrain(i+j-1)==0 && noOfZeros<2
                        noOfZeros=noOfZeros+1;
                        k=k+1;
                        balancedFeaturesTrain(k,:)=featuresTrain(i+j-1,:);
                        balancedclassesTrain(k)=classesTrain(i+j-1);
                    end
                end
            end
            featuresTrain=balancedFeaturesTrain;
            classesTrain=balancedclassesTrain;
            
        elseif balancingFlag==2
            %Balancing the training data set by addin 6 noisy trials to each Target trial
            goodTrain=0;
            for i=1:length(classesTrain)
                goodTrain=goodTrain+1;
                balancedFeaturesTrain(goodTrain,:)=featuresTrain(i,:);
                balancedclassesTrain(goodTrain)=classesTrain(i);
                if classesTrain(i)==1
                    for new=1:6
                        goodTrain=goodTrain+1;
                        balancedFeaturesTrain(goodTrain,:)=awgn(featuresTrain(i,:),10,'measured');
                        balancedclassesTrain(goodTrain)=classesTrain(i);
                    end
                end
            end
            featuresTrain=balancedFeaturesTrain;
            classesTrain=balancedclassesTrain;
        end
        
        %Feature Selection (0.5% provides 95% acc on test set
        numberOfFeatures=fix(percentageOfFeatures*size(featuresTrain,2));
        [B,Info]=lasso(featuresTrain, classesTrain','Alpha',1,'cv',10,'DFmax',numberOfFeatures);
        featuresVector=[];
        for i=1:size(featuresTrain,2)
            if sum(B(i,:))~=0
                featuresVector=[featuresVector; i];
            end
        end
        
        featuresTrain=featuresTrain(:,featuresVector);
        featuresTest=featuresTest(:,featuresVector);
        
        classifierTrainClasses=classify(featuresTrain,featuresTrain,classesTrain,'Linear');
        results=confusionmat(classifierTrainClasses,classesTrain);
        accTrain=trace(results)/sum(sum(results));
        
        classifierTestClasses=classify(featuresTest,featuresTrain,classesTrain,'Linear');
        results=confusionmat(classifierTestClasses,classesTest);
        accTest=trace(results)/sum(sum(results));
        
        goodTrain=0;
        for i=1:16:length(classifierTrainClasses)
            for j=1:16
                flag=1;
                if classifierTrainClasses(i+j-1)~=classesTrain(i+j-1)
                    flag=0;
                    break
                end
            end
            if flag==1
                goodTrain=goodTrain+1;
            end
        end
        goodTrain=goodTrain/(length(classifierTrainClasses)/16);
        
        goodTest=0;
        for i=1:16:length(classifierTestClasses)
            for j=1:16
                flag=1;
                if classifierTestClasses(i+j-1)~=classesTest(i+j-1)
                    flag=0;
                    break
                end
            end
            if flag==1
                goodTest=goodTest+1;
            end
        end
        goodTest=goodTest/(length(classifierTestClasses)/16);
        
        resultsMatrix(subject,:)=[accTrain accTest goodTrain goodTest] ;
        
        % %     fileNameCSV=['dataMatrix' fileName '.csv'];
        % %     writematrix(dataMatrix,fileNameCSV);
        % %     fileNameTargetCSV=['targetsVector' fileName '.csv'];
        % %     writematrix(targetsVector,fileNameTargetCSV);
        
    end
    
    resultsMatrix
end
%mean(resultsMatrix)