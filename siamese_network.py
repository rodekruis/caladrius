import numpy as np

import torch
import torch.nn as nn
import torch.functional as F


def create_model(conv_layers_parameters):
    layers = []
    for conv_spec in conv_layers_parameters:
        layers.append(nn.Conv2d(**conv_spec))
        layers.append(nn.BatchNorm2d(conv_spec['out_channels']))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))

    return nn.Sequential(*layers)


def create_twins(conv_layers_parameters, share_instance: bool = False):
    twin = create_model(conv_layers_parameters)

    if share_instance:
        return twin, twin
    else:
        twin2 = create_model(conv_layers_parameters)
        return twin, twin2

    return twin, twin2


def create_sequential_for_twin(input_dim, n_classes):
    fc = nn.Sequential(
        nn.Linear(input_dim, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(512, n_classes),
        nn.Softmax(0)
    )
    return fc


class SiameseNet(nn.Module):
    def __init__(self, twins, sequential):
        super().__init__()
        self.twin1, self.twin2 = twins
        self.fc1 = sequential

    def get_conv_output_shape(self, x):
        return self.twin1(x).shape

    def forward(self, x1, x2):
        out1 = self.twin1(x1)
        out2 = self.twin2(x2)

        combined = torch.cat([out1, out2], 1)
        combined = combined.view(combined.size(0), -1)
        return self.fc1(combined)


def build_net(image_size, n_classes, in_channels=3):

    conv_layers_parameters = []

    conv_layers_parameters.append(
        dict(in_channels=in_channels, out_channels=64, kernel_size=3))
    conv_layers_parameters.append(dict(
        in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=128, kernel_size=3))
    conv_layers_parameters.append(dict(
        in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=128, kernel_size=3))
    conv_layers_parameters.append(dict(
        in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=256, kernel_size=3))
    conv_layers_parameters.append(dict(
        in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=256, kernel_size=3))
    conv_layers_parameters.append(dict(
        in_channels=conv_layers_parameters[-1]['out_channels'], out_channels=256, kernel_size=3))

    twins = create_twins(conv_layers_parameters)

    dummy_input = torch.zeros(1, in_channels, *image_size)

    out = twins[0](dummy_input)
    out_size = out.size()[1:]
    out_size = np.prod([a for a in out_size])

    sequential = create_sequential_for_twin(
        input_dim=out_size*2, n_classes=n_classes)

    return SiameseNet(twins, sequential)


def get_transforms(set_name, input_shape):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    train_transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    return {
        "train": train_transform,
        "val": test_transform,
        "test": test_transform
    }[set_name]


if __name__ == '__main__':
    net = build_net((299, 299), 3)
    print(net)
    out = net(torch.zeros(3, 3, 299, 299), torch.zeros(3, 3, 299, 299))
    print(out.size())
