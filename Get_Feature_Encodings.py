import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torchvision.transforms.functional as TF
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


def ReturnFeatureEmbeddings(encoding_dir):
    # Extract [w, H(w)] from directory
    encodings_numpy = np.load(encoding_dir)

    # Create empty torch tensors 
    w_encodings = torch.zeros((len(encodings_numpy), encodings_numpy[0][0].shape[0]))
    hw_encodings = torch.zeros((len(encodings_numpy), encodings_numpy[0][1].shape[0]))

    # Iterate through encodings, cast them to torch tensors
    for i in range(len(encodings_numpy)):
        w_encodings[i] = torch.from_numpy(encodings_numpy[i][0])
        hw_encodings[i] = torch.from_numpy(encodings_numpy[i][1])

    # Return data loader
    encoding_loader = datamanager.TensorToDataLoader(xData = hw_encodings, yData = w_encodings, batchSize = 64)
    return encoding_loader


def CreateIdentityFeatureEmbeddings(encoding_dir):
    # Extract [w, H(w)] from directory
    encodings_numpy = np.load(encoding_dir)

    # Create empty torch tensors 
    w_encodings = torch.zeros((len(encodings_numpy), encodings_numpy[0][0].shape[0]))

    # Create randomized list of indices
    indexList = []
    for i in range(0, len(encodings_numpy)):
        indexList.append(i)
    shuffle(indexList)

    # Iterate through encodings, cast them to torch tensors
    for i in range(len(encodings_numpy)):
        w_encodings[i] = torch.from_numpy(encodings_numpy[indexList[i]][0])

    # Return data loader
    encoding_loader = datamanager.TensorToDataLoader(xData = w_encodings, yData = w_encodings, batchSize = 64)
    return encoding_loader


def CreateRandomConcatenatedFeatureEmbeddings(encoding_dir):
    # Extract [w, H(w)] from directory
    encodings_numpy = np.load(encoding_dir)

    # Create random 1024 x 1 vector for concatetating w
    H = np.random.choice([0, 1], size=(1024, 1))

    # Create empty torch tensors 
    w_encodings = torch.zeros((len(encodings_numpy), encodings_numpy[0][0].shape[0]))
    hw_encodings = torch.zeros((len(encodings_numpy), encodings_numpy[0][0].shape[0] + 1024))

    # Iterate through encodings, cast them to torch tensors
    for i in range(len(encodings_numpy)):
        w_encodings[i] = torch.from_numpy(encodings_numpy[i][0])
        #print(w_encodings[i].size)
        #print(encodings_numpy[i][1].shape)
        temp_hw_encoding = torch.from_numpy(encodings_numpy[i][1]).unsqueeze(dim = 1)
        temp_hw_encoding = torch.cat(tensors = (temp_hw_encoding, torch.from_numpy(H)), dim = 0)
        #print(temp_hw_encoding.size())
        hw_encodings[i] = temp_hw_encoding.squeeze(dim = 1)
        #print(hw_encodings[i].size)

    # Return data loader
    encoding_loader = datamanager.TensorToDataLoader(xData = hw_encodings, yData = w_encodings, batchSize = 64)
    return encoding_loader


def Return_ldpc_Embeddings(encoding_dir):
    # Open the file in read mode
    ldpc_encoding_list = open(encoding_dir, "r") 
  
    # Read file, split into [w, Hw]
    ldpc_encoding_list = ldpc_encoding_list.read()
    ldpc_encoding_list = ldpc_encoding_list.split(", ")
    #print(ldpc_encoding_list)

    # For each [w, Hw], convert to formal list
    w_encodings = []
    hw_encodings = []
    cur_string_encoding = []
    cur_string_read = 0
    cur_w_read = True
    for string_encoding in ldpc_encoding_list:
        if cur_w_read:
            if '0' in string_encoding: cur_string_encoding.append(0)
            if '1' in string_encoding: cur_string_encoding.append(1)
            cur_string_read += 1
            if cur_string_read == 1024:
                w_encodings.append(cur_string_encoding)
                cur_string_encoding = []
                cur_string_read = 0 
                cur_w_read = False
        else:
            if '0' in string_encoding: cur_string_encoding.append(0)
            if '1' in string_encoding: cur_string_encoding.append(1)
            cur_string_read += 1
            if cur_string_read == 512:
                hw_encodings.append(cur_string_encoding)
                cur_string_encoding = []
                cur_string_read = 0 
                cur_w_read = True
  
    # Create empty torch tensors 
    w_encodings_torch = torch.zeros((len(w_encodings), len(w_encodings[0])))
    hw_encodings_torch = torch.zeros((len(hw_encodings), len(hw_encodings[0])))

    # Iterate through encodings, cast them to torch tensors
    for i in range(len(w_encodings)):
        w_encodings_torch[i] = torch.Tensor(w_encodings[i])
        hw_encodings_torch[i] = torch.Tensor(hw_encodings[i])
        #print(w_encodings_torch[i])

    # Return data loader
    encoding_loader = datamanager.TensorToDataLoader(xData = hw_encodings_torch, yData = w_encodings_torch, batchSize = 64)
    return encoding_loader

if __name__ == '__main__':
    Return_ldpc_Embeddings(os.getcwd() + '//ldpc_embeddings.txt')