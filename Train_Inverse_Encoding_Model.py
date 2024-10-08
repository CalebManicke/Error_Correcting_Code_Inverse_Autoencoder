import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchsummary import summary
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
from random import shuffle
import Data_Manager as datamanager
import os
from PIL import Image
from random import shuffle

from Get_Feature_Encodings import ReturnFeatureEmbeddings, CreateIdentityFeatureEmbeddings, CreateRandomConcatenatedFeatureEmbeddings, Return_ldpc_Embeddings, Return_Big_ldpc_Embeddings
from Inverse_Encoding_Model import Linear_Hw_To_w_Model, One_Conv_Inverse_Hw_To_w_Model, Conv_Inverse_Hw_To_w_Autoencoder, Inverse_Hw_To_w_DenseNet

# Train network
def train(numEpochs, model, trainLoader, valLoader, device, continueTraining, optimizer, criterion, scheduleList, saveTag, saveDir):
    # Save model with highest val acc at certain epoch
    bestModel = None
    bestEpoch = 0
    bestValAcc = 0
    curEpoch = 1

    if continueTraining:
        checkpoint = torch.load(saveDir + "/" + saveTag + ".th", map_location = torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        bestModel = checkpoint['best_state_dict']
        bestValAcc = checkpoint['bestValAcc']
        bestEpoch = checkpoint['bestEpoch']
        curEpoch = checkpoint['curEpoch']
        optimizer = checkpoint['optimizer']
        criterion = checkpoint['criterion']
        print("Resuming training from ", curEpoch, "th epoch...")

    # Start training
    print("------------------------------------")
    print("Training Starting...")
    preTrainValAcc = datamanager.validateReturn(model.to(device), valLoader, device)
    print("Before Training Val Acc: ", preTrainValAcc)
    print("------------------------------------")

    for epoch in range(curEpoch, numEpochs+1):
        model.train()
        #print("Training Epoch: ", epoch)
        avg_loss = 0
        for i, (data, targets) in enumerate(trainLoader):
            # Should be 1 if identity...
            #print(torch.min(torch.eq(data, targets).int()))
            #print(torch.eq(data, targets))

            data = Variable(data.to(device = device), requires_grad = True)
            #targets = targets.to(device) # .float()
            target_vars = targets.to(device) #.long().targets.unsqueeze(1)

            # Forward pass
            scores = model(data).to(device)
            #scores = Variable(scores.to(device = device), requires_grad = True)
            loss = criterion(scores, target_vars)   # Loss function
            avg_loss += loss.item() /len(trainLoader)

            # Backward
            optimizer.zero_grad() # We want to set all gradients to zero for each batch so it doesn't store backprop calculations from previous forwardprops
            loss.backward()
            optimizer.step()

        for scheduler in scheduleList:     scheduler.step()
        currentLR = optimizer.param_groups[-1]['lr']
        print("Current Learning Rate: ", currentLR)
        print("Average Loss: ", avg_loss)
        trainLoader = datamanager.ManuallyShuffleDataLoader(trainLoader)

        # For every 20 epochs, print validation accuracy and save model with higher val acc
        if epoch % 10 == 0:
            valAcc = datamanager.validateReturn(model, valLoader, device)
            #trainLoader = datamanager.ManuallyShuffleDataLoader(trainLoader)
            trainAcc = datamanager.validateReturn(model, trainLoader, device)
            if valAcc > bestValAcc:
                bestModel = model.state_dict()
                bestEpoch = epoch
                bestValAcc = valAcc
            print("------------------------------------")
            print("Current Epoch: ", epoch)
            print("Validation Accuracy: ", valAcc)
            print("Training Accuracy: ", trainAcc)
            print("Best Epoch: ", bestEpoch)
            print("Best Val Acc: ", bestValAcc)
            print("------------------------------------")
            torch.save({'curEpoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer, 'criterion': criterion, 'best_state_dict': bestModel,  'bestEpoch': bestEpoch, 'bestValAcc': bestValAcc}, os.path.join(saveDir, saveTag+'.th'))

    print("------------------------------------")
    print("Training Completed...")
    print("------------------------------------")
    return bestModel, bestEpoch, bestValAcc


def main(train_dict, encoding_dir):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: ', device)
    if (torch.cuda.is_available()):
        print('Number of CUDA Devices:', torch.cuda.device_count())
        print('CUDA Device Name:',torch.cuda.get_device_name(0))
        print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)
        
    # Create folders for trained models 
    save_dir = os.getcwd() + "//Trained_Hw_To_w_Models//"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    # If dataloaders haven't been created yet, call ReturnVoterLabDataLoaders
    if 'ldpc' in train_dict['modelName']: encoding_loader = Return_ldpc_Embeddings(encoding_dir)
    if 'Big_ldpc' in train_dict['modelName']: encoding_loader = Return_Big_ldpc_Embeddings(train_dict['batchSize'])
    if 'Identity' in train_dict['modelName']: encoding_loader = CreateIdentityFeatureEmbeddings(encoding_dir)
    if 'Concat' in train_dict['modelName']: encoding_loader = CreateRandomConcatenatedFeatureEmbeddings(encoding_dir)
    if 'Random_Mult' in train_dict['modelName']: encoding_loader = ReturnFeatureEmbeddings(encoding_dir)
    
    # Randomly perform 80-20% training/validation split 
    encoding_loader = datamanager.ManuallyShuffleDataLoader(encoding_loader)
    hw_encodings, w_encodings = datamanager.DataLoaderToTensor(encoding_loader)
    num_training_encodings = int(len(hw_encodings) * 0.95)
    train_encoding_loader = datamanager.TensorToDataLoader(xData = hw_encodings[:num_training_encodings], yData = w_encodings[:num_training_encodings], batchSize = train_dict['batchSize'])
    val_encoding_loader = datamanager.TensorToDataLoader(xData = hw_encodings[num_training_encodings:], yData = w_encodings[num_training_encodings:], batchSize = train_dict['batchSize'])
    print("Training Encoding Loader Length: " + str(len(train_encoding_loader)))
    print("Validation Encoding Loader Length: " + str(len(val_encoding_loader)))

    # Initialize models, store in dictionary
    hw_shape, w_shape = datamanager.GetLoaderShape(encoding_loader)
    if train_dict['modelType'] == 'linear': model = Linear_Hw_To_w_Model(insize = hw_shape[0], outsize = w_shape[0])
    if train_dict['modelType'] == 'autoencoder': model = Conv_Inverse_Hw_To_w_Autoencoder()
    if train_dict['modelType'] == 'DenseNet': model = Inverse_Hw_To_w_DenseNet(dropout_rate = 1.0, out_features = 1024)

    # Load pretrained weights if specified
    if not (train_dict['usePretrained'] is None):
        with torch.no_grad():
            checkpointLocation = train_dict['usePretrained']
            checkpoint = torch.load(checkpointLocation, map_location = torch.device("cpu"))
            model.load_state_dict(checkpoint['state_dict'])

        if 'ldpc' in train_dict['modelName']:
            big_encoding_loader = Return_Big_ldpc_Embeddings(train_dict['batchSize'])
            valAcc = datamanager.validateReturn(model.to(device), big_encoding_loader, device)
            print("------------------------------------")
            print("Total Pre-Trained Accuracy: ", valAcc)

    # Train model
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr = train_dict['learningRate'], weight_decay = train_dict['weightDecay'])
    train_dict['lrScheduler'] = [] #[torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98), torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,40, 60,80,100,120,140,160,180], gamma=1.2)] # gamma=1.2
    print("------------------------------------")

    # Train and validate
    saveTag = train_dict['modelName']
    saveDir = save_dir
    bestModel, bestEpoch, bestValAcc = train(numEpochs = train_dict['numEpochs'], model = model, trainLoader = train_encoding_loader, valLoader = val_encoding_loader, device = device, continueTraining = train_dict['continueTraining'], optimizer = optimizer, criterion = criterion, scheduleList = train_dict['lrScheduler'], saveTag = saveTag, saveDir = saveDir)
    model.eval()
    print("------------------------------------")
    print("Done training " + train_dict['modelName'] + "...")
    trainAcc = datamanager.validateReturn(model.to(device), train_encoding_loader, device)
    valAcc = datamanager.validateReturn(model.to(device), val_encoding_loader, device)
    print("Final Training Accuracy: ", trainAcc)
    print("Final Validation Accuracy: ", valAcc)
    print("------------------------------------")
        
    # Save trained model
    torch.save({'numEpoch': train_dict['numEpochs'], 'state_dict': model.state_dict(), 'valAcc': valAcc, 'trainAcc': trainAcc, 'best_state_dict': bestModel, 'bestEpoch': bestEpoch, 'bestValAcc': bestValAcc}, os.path.join(saveDir, saveTag + '.pth')) 

if __name__ == '__main__':
    # Random_Linear_Hw_To_w_Model = {'modelName': 'Random_Mult', 'learningRate': 0.00001, 'weightDecay': 1e-2, 'numEpochs': 200, 'lrScheduler': [], 'continueTraining': False, 'batchSize': 4}
    trainLDPC = True 

    # Working configuration: {'modelName': 'Identity_DenseNet', 'learningRate': 0.00001, 'weightDecay': 0, 'numEpochs': 200, 'lrScheduler': [], 'continueTraining': False, 'batchSize': 4}, model droupout_rate = 1.0, no learning rate scheduler!
    Random_Linear_Hw_To_w_Model = dict()
    pretrained_dir = os.getcwd() + '//Trained_Hw_To_w_Models//Big_ldpc_DenseNet.th'
    if trainLDPC: Random_Linear_Hw_To_w_Model = {'modelName': 'ldpc_DenseNet', 'modelType': 'DenseNet', 'dropoutRate': 1, 'learningRate': 0.00001, 'weightDecay': 0, 'numEpochs': 200, 'lrScheduler': [], 'continueTraining': False, 'batchSize': 32, 'usePretrained': pretrained_dir}
    else:         Random_Linear_Hw_To_w_Model = {'modelName': 'Identity_DenseNet', 'modelType': 'DenseNet', 'dropoutRate': 1, 'learningRate': 0.0001, 'weightDecay': 0, 'numEpochs': 200, 'lrScheduler': [], 'continueTraining': False, 'batchSize': 64, 'usePretrained': pretrained_dir}

    # Linear model: learning rate = 1e-5, batch size = 16, weight decay = 0
    if 'Identity' in Random_Linear_Hw_To_w_Model['modelName']: encoding_dir = os.getcwd() + '//lord_caleb.npy'
    if 'ldpc' in Random_Linear_Hw_To_w_Model['modelName']: encoding_dir = os.getcwd() + '//ldpc_sparse.txt'
    if 'Synthetic_ldpc' in Random_Linear_Hw_To_w_Model['modelName']: encoding_dir = os.getcwd() + '//synthetic_ldpc_embeddings.txt'
    main(Random_Linear_Hw_To_w_Model, encoding_dir)