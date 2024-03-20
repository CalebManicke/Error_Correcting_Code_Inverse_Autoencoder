import torch 
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math 
import random 
#import matplotlib.pyplot as plt
#import os 
#import PIL
from random import shuffle

#Class to help with converting between dataloader and pytorch tensor 
class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, x_tensor, y_tensor, transforms=None):
        self.x = x_tensor
        self.y = y_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        if self.transforms is None: #No transform so return the data directly
            return (self.x[index], self.y[index])
        else: #Transform so apply it to the data before returning 
            return (self.transforms(self.x[index]), self.y[index])

    def __len__(self):
        return len(self.x)

#Validate using a dataloader 
def validateD(valLoader, model, device=None):
    #switch to evaluate mode
    model.eval()
    acc = 0 
    batchTracker = 0
    with torch.no_grad():
        #Go through and process the data in batches 
        for i, (input, target) in enumerate(valLoader):
            sampleSize = input.shape[0] #Get the number of samples used in each batch
            batchTracker = batchTracker + sampleSize
            #print("Processing up to sample=", batchTracker)
            if device == None: #assume cuda
                inputVar = input.cpu() #.cuda()
            else:
                inputVar = input.to(device)
            #compute output
            output = model(inputVar)
            output = output.float()
            #Go through and check how many samples correctly identified
            for j in range(0, sampleSize):
                if output[j].argmax(axis=0) == target[j]:
                    acc = acc +1
    acc = acc / float(len(valLoader.dataset))
    return acc

#Convert a X and Y tensors into a dataloader
#Does not put any transforms with the data  
def TensorToDataLoader(xData, yData, transforms= None, batchSize=None, randomizer = None):
    if batchSize is None: #If no batch size put all the data through 
        batchSize = xData.shape[0]
    dataset = MyDataSet(xData, yData, transforms)
    if randomizer == None: #No randomizer
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, shuffle=False)
    else: #randomizer needed 
        train_sampler = torch.utils.data.RandomSampler(dataset)
        dataLoader = torch.utils.data.DataLoader(dataset=dataset,  batch_size=batchSize, sampler=train_sampler, shuffle=False)
    return dataLoader

#Convert a dataloader into x and y tensors 
def DataLoaderToTensor(dataLoader):
    #First check how many samples in the dataset
    numSamples = len(dataLoader.dataset) 
    inputShape, outputShape = GetLoaderShape(dataLoader) #Get the output shape from the dataloader
    sampleIndex = 0
    #xData = torch.zeros(numSamples, sampleShape[0], sampleShape[1], sampleShape[2])
    xData = torch.zeros((numSamples,) + inputShape) #Make it generic shape for non-image datasets
    yData = torch.zeros((numSamples,) + outputShape)
    #Go through and process the data in batches 
    for i, (input, target) in enumerate(dataLoader):
        batchSize = input.shape[0] #Get the number of samples used in each batch
        #Save the samples from the batch in a separate tensor 
        for batchIndex in range(0, batchSize):
            xData[sampleIndex] = input[batchIndex]
            yData[sampleIndex] = target[batchIndex]
            sampleIndex = sampleIndex + 1 #increment the sample index 
    return xData, yData 

#Get xData and yData dataloader sizes
def GetLoaderShape(dataLoader):
    for i, (input, target) in enumerate(dataLoader):
        return input[0].shape, target[0].shape

#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled

# Outputs accuracy given data loader and inverse feature encoder
def validateReturn(model, loader, device):
    model.eval()
    numCorrect = 0
    batchTracker = 0
    # Without adding to model loss, go through each batch, compute output, tally how many examples our model gets right
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            target = target.to(device)
            #print('Target:', target)
            data = data.to(device)
            output = model(data.to(device))
            #print("Output: ", output)
            binary_output = torch.zeros(output.shape)
            # Output is continuous, need to map to bit string to find Hamming distance
            binary_output = torch.gt(output, 0.5).int().to(device)
            #print("Binary: ", binary_output)
            #(torch.sum(data, dim = 1) >= 0.5).int()
            '''
            for encoding_index in range(0, output.shape[0]):
                for cont_index in range(0, output.shape[1]):
                    #print(output[encoding_index][cont_index])
                    if output[encoding_index][cont_index] >= 0:   binary_output[encoding_index][cont_index] = 1
                    if output[encoding_index][cont_index] <  0:   binary_output[encoding_index][cont_index] = 0
            '''
            #print(torch.max(binary_output))
            binary_output = binary_output.to(device)
            # Compute Hamming distance 
            '''
            for encoding_index in range(0, output.shape[0]):
                cur_hamming_dist = 0
                for bit_index in range(0, output.shape[1]):
                    if int(output[encoding_index][cont_index]) == int(binary_output[encoding_index][cont_index]): cur_hamming_dist += 1
                numCorrect += (cur_hamming_dist / output.shape[1])
            '''
            batch_hamming_dist = torch.sum(torch.eq(target, binary_output).int(), dim = 1) / int(binary_output.size(dim = 1))
            #print(batch_hamming_dist)
            avg_hamming_dist = torch.sum(batch_hamming_dist, dim = 0)
            #print(avg_hamming_dist)
            numCorrect += avg_hamming_dist
    # Compute raw accuracy
    acc = numCorrect.item() / float(len(loader.dataset))
    return acc