imds=imageDatastore('E:\SS\Data','IncludeSubfolders',true,'LabelSource','foldernames');
img=readimage(imds,1);
size(img);
imd=imageDatastore('E:\SS\Test','IncludeSubfolders',true,'LabelSource','foldernames');
trainingSet=imds;
testSet=imd;
layers = [
    imageInputLayer([576 576 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',testSet, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
net = trainNetwork(trainingSet,layers,options);
YPred = classify(net,testSet);
YPred
YValidation = testSet.Labels;

accuracy = sum(YPred == YValidation)/numel(YValidation)
