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

import re
from collections import OrderedDict
from functools import partial
from typing import Any, List, Optional, Tuple
import torch.utils.checkpoint as cp
from torch import Tensor

# Create model loader 
class Linear_Hw_To_w_Model(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        # Four layers, number of features grows linearly
        growth_rate = int((outsize - insize) // 4) + int(outsize // 4)
        print('Growth Rate: ' + str(growth_rate))
        #self.conv2dtrans_1 = torch.nn.ConvTranspose2d(insize, (1.5)*insize, (insize/256))
        #self.conv2dtrans_2 = torch.nn.ConvTranspose2d((1.5)*insize, (2.5)*insize, (insize/256))
        self.forward0 = torch.nn.Linear(in_features = insize, out_features = insize + growth_rate, bias = True)
        self.forward1 = torch.nn.Linear(in_features = insize + growth_rate, out_features = insize + 2 * growth_rate, bias = True)
        self.forward2 = torch.nn.Linear(in_features = insize + 2 * growth_rate, out_features = insize + 3 * growth_rate, bias = True)
        self.forward3 = torch.nn.Linear(in_features = insize + 3 * growth_rate, out_features = insize + growth_rate, bias = True)
        self.forward4 = torch.nn.Linear(in_features = insize + growth_rate, out_features = outsize, bias = True)
        #self.softmax = torch.nn.Softmax(dim = 0)

    def forward(self, x):
        #out = self.conv2dtrans_1(x)
        #out = self.conv2dtrans_2(out)
        out = self.forward0(x)
        out = self.forward1(out)
        out = self.forward2(out)
        out = self.forward3(out)
        out = self.forward4(out)
        return F.sigmoid(out) #self.softmax(out)

# Model consisting of only one convolutional layer: we only need to learn one inverse matrix multiplication... this is just one convolutional block!
class One_Conv_Inverse_Hw_To_w_Model(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()
        # Create one convolutional block
        self.conv_matrices = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.Conv2d(1, 1, 1) ),
                    ("conv1", nn.Conv2d(1, 1, 1) ),
                    ("conv2", nn.Conv2d(1, 1, 1) ),
                    ("conv3", nn.Conv2d(1, 1, 1) )
                ])
            )
        conv_output_shape = torch.flatten(self.conv_matrices(torch.zeros((1, 1, 32, 32)))) 
        print('Conv Length Encoding: ', len(conv_output_shape))
        self.forward0 = nn.Linear(in_features = len(conv_output_shape), out_features = outsize, bias = True)

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        resized_data = torch.zeros((x.size(dim = 0), 1, 32, 32))
        for i in range(x.size(dim = 0)):
            resized_data[i][0] = TF.resize(img = x[i].unsqueeze(dim = 0).unsqueeze(dim = 0), size = (32, 32))
            print(resized_data[i].size())
        resized_data = resized_data.to(device)

        out = self.conv_matrices(resized_data)
        out = torch.flatten(out)
        print(out.size())
        return F.sigmoid(self.forward0(out))


class Conv_Inverse_Hw_To_w_Autoencoder(nn.Module):
    def __init__(self):
        super(Conv_Inverse_Hw_To_w_Autoencoder, self).__init__()
        # Encoder layers

        # Encoders/Decoders want to blur... we want to add noise! We need less parameters...
        # We are loosing too much information in the info space! We cannot have too many convolutions/pooling layers in inverse denoiser...
        self.inverse_encoder = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.Conv2d(1, 32, 3, padding=1) ),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("pool0", nn.MaxPool2d(2, 2)), 
                    ("conv1", nn.Conv2d(32, 16, 3, padding=1) ),
                    ("relu1", nn.ReLU(inplace = True)),
                    ("pool1", nn.MaxPool2d(2, 2)),
                    ("conv2", nn.Conv2d(16, 8, 3, padding=1) ),
                    ("relu2", nn.ReLU(inplace = True)),
                    ("pool2", nn.MaxPool2d(2, 2))
                ])
            )
        
        # Decoder layers
        self.inverse_decoder = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.ConvTranspose2d(8, 8, 3, stride=2) ),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("conv1", nn.ConvTranspose2d(8, 16, 2, stride=2) ),  #2, stride=2
                    ("relu1", nn.ReLU(inplace = True)),
                    ("conv2", nn.ConvTranspose2d(16, 32, 2, stride=2) ),  #2, stride=2
                    ("relu2", nn.ReLU(inplace = True)),
                    ("conv3", nn.Conv2d(32, 1, 3, padding=0))
                ])
            )

    def forward(self, x):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        resized_data = torch.zeros((x.size(dim = 0), 1, 32, 32))
        for i in range(x.size(dim = 0)):
            resized_data[i][0] = TF.resize(img = x[i].unsqueeze(dim = 0).unsqueeze(dim = 0), size = (32, 32))
        resized_data = resized_data.to(device)

        out = self.inverse_encoder(resized_data)
        #print(out.size())
        out = self.inverse_decoder(out)

        # Our decoder produces 42 x 50 images, to match we nick the first and bottom row
        correctDimOut = torch.zeros((out.size(dim = 0), 1024))
        #print("Output Size: ", out.size())
        for i in range(out.size(dim = 0)):
            curImage = out[i][0]
            curImage = curImage[1:]
            curImage = curImage[:curImage.size(dim = 0) - 1]
            curImage = curImage[:, 1:]
            curImage = curImage[:, :curImage.size(dim = 1) - 1]
            #print("Output Size: ", curImage.size())
            correctDimOut[i] = torch.flatten(curImage)

        return F.sigmoid(correctDimOut)
        #return torch.gt(correctDimOut, 0).int().float() #.to(device)


# Built-in DenseNet architecture functions, sourcecode found here: https://pytorch.org/vision/main/_modules/torchvision/models/densenet.html#densenet169
class _DenseLayer(nn.Module):
    def __init__(
        self, num_input_features: int, growth_rate: int, bn_size: int, drop_rate: float, memory_efficient: bool = False
    ) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False
    
    @torch.jit.unused  # noqa: T484
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: List[Tensor]) -> Tensor:  # noqa: F811
        pass

    @torch.jit._overload_method  # noqa: F811
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        pass

    # torchscript does not yet support *args, so we overload method
    # allowing it to take either a List[Tensor] or single Tensor
    def forward(self, input: Tensor) -> Tensor:  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ) -> None:
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

# We modify main DenseNet function by introducing more FC layers at the end and 0.5 Dropout
class _Inverse_Hw_To_w_DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        No num_classes since we're performing binary classification
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def  __init__(
        self,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int, int] = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0,
        input_channel: int = 3,
        final_dropout_rate: float = 0.5,
        memory_efficient: bool = False,
        out_features: int = 1,
    ) -> None:

        # Voterlab data is (3, 40, 50) so we don't use 5 pixel up/down padding
        super().__init__()
        #_log_api_usage_once(self)

        # First convolution
        self.features = nn.Sequential(
            OrderedDict(
                [
                    ("conv0", nn.Conv2d(input_channel, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        # Linear layer: 1024 units --> 1 unit
        self.Dropout = nn.Dropout(p = final_dropout_rate)
        self.forward0 = nn.Linear(in_features = num_features, out_features = out_features, bias = False)

        '''
        # Decoder layers
        self.inverse_decoder = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.ConvTranspose2d(8, 8, 3, stride=2) ),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("conv1", nn.ConvTranspose2d(8, 16, 2, stride=2) ),  #2, stride=2
                    ("relu1", nn.ReLU(inplace = True)),
                    ("conv2", nn.ConvTranspose2d(16, 32, 2, stride=2) ),  #2, stride=2
                    ("relu2", nn.ReLU(inplace = True)),
                    ("conv3", nn.Conv2d(32, 1, 3, padding=0))
                    #("conv0", nn.Conv2d(3, 1, 3, padding=1) ),
                    #("relu0", nn.ReLU(inplace = True)),
                    #("pool0", nn.MaxPool2d(2, 2))
                ])
            )
        '''

    def forward(self, x: Tensor) -> Tensor:
        #print(x.size())
        #x = x.cuda()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        resized_data = torch.zeros((x.size(dim = 0), 1, 32, 32))
        for i in range(x.size(dim = 0)):
            resized_data[i][0] = TF.resize(img = x[i].unsqueeze(dim = 0).unsqueeze(dim = 0), size = (32, 32))
        resized_data = resized_data.to(device)

        features = self.features(resized_data)
        out = F.relu(features, inplace=True)
        # This is global average pooling since we set output size to (1, 1)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        #out = self.Dropout(out)
        # 1024 unit FC --> ReLU --> 1 unit FC --> Sigmoid
        #out = F.relu(self.forward0(out))
        out = F.sigmoid(self.forward0(out))
        return out
    
    
# Return Denoising DenseNet
def Inverse_Hw_To_w_DenseNet (dropout_rate, out_features): 
    return _Inverse_Hw_To_w_DenseNet(growth_rate = 32, block_config =  (6, 12, 24, 16), input_channel = 1, final_dropout_rate = dropout_rate, num_init_features = 64, out_features = out_features)