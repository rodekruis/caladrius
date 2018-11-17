# https://github.com/delijati/pytorch-siamese/blob/master/net.py
# %%
import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

# %%
input_dim = dict(image=32, fully=11)
# input_dim = dict(image=256, fully=30)

conv_layers_parameters = []
in_channels = 1
conv_layers_parameters.append(dict(in_channels=in_channels, out_channels=64, kernel_size=10))
conv_layers_parameters.append(dict(in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=128, kernel_size=7))
conv_layers_parameters.append(dict(in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=128, kernel_size=4))
conv_layers_parameters.append(dict(in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=256, kernel_size=4))

# %%
def create_model(conv_layers_parameters):
    layers = []
    for conv_spec in conv_layers_parameters:
        layers.append(nn.Conv2d(**conv_spec))
        layers.append(nn.BatchNorm2d(conv_spec['out_channels']))
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def create_twins(conv_layers_parameters):
    twin1 = create_model(conv_layers_parameters)
    twin2 = create_model(conv_layers_parameters)

    return twin1, twin2


def create_sequential_for_twin(input_dim, nclasses):
    fc = nn.Sequential(
        nn.Linear(input_dim, 32),
        nn.ReLU(inplace=True),
        nn.Linear(32, nclasses),
        nn.Softmax()
    )
    return fc


# %%
class SiameseNet(nn.Module):
    def __init__(self, twins, sequential):
        super().__init__()
        self.twin1, self.twin2 = twins
        self.fc1 = sequential

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def get_conv_output_shape(self, x):
        return self.twin1(x).shape

    def forward(self, x: tuple):
        out1 = self.twin1(x[0])
        out2 = self.twin2(x[1])
        combined = torch.cat([out1, out2])

        combined = combined.view(combined.size(0), -1)
        out = self.fc1(combined)
        return out


# %%
twins = create_twins(conv_layers_parameters)
sequential = create_sequential_for_twin(input_dim=11*11*conv_layers_parameters[-1]['out_channels'], nclasses=4)

s = SiameseNet(twins, sequential)

data1 = torch.zeros(32, 1, input_dim['image'], input_dim['image'])
data2 = torch.zeros((32, 1, input_dim['image'], input_dim['image']))
data = (data1, data2)

s(data)
