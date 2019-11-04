import time
import copy
from collections import OrderedDict

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from utils import create_logger


logger = create_logger(__name__)


def get_pretrained_iv3(output_size):
    model_conv = torchvision.models.inception_v3(pretrained=True)

    for i, param in model_conv.named_parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, output_size)

    ct = []
    for name, child in model_conv.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    # To view which layers are freeze and which layers are not freezed:
    for name, child in model_conv.named_children():
        for name_2, params in child.named_parameters():
            logger.debug("{}, {}".format(name_2, params.requires_grad))

    return model_conv


def get_pretrained_iv3_transforms(set_name):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    input_shape = 299
    train_transform = transforms.Compose(
        [
            transforms.Resize(scale),
            transforms.RandomResizedCrop(input_shape),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=90),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(scale),
            transforms.CenterCrop(input_shape),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return {
        "train": train_transform,
        "validation": test_transform,
        "test": test_transform,
    }[set_name]


class SiameseNetwork(nn.Module):
    def __init__(
        self, output_size=512, similarity_layers_sizes=[512, 512], dropout=0.5
    ):
        super().__init__()
        self.left_network = get_pretrained_iv3(output_size)
        self.right_network = get_pretrained_iv3(output_size)

        similarity_layers = OrderedDict()
        similarity_layers["layer_0"] = nn.Linear(
            output_size * 2, similarity_layers_sizes[0]
        )
        similarity_layers["relu_0"] = nn.ReLU(inplace=True)
        similarity_layers["bn_0"] = nn.BatchNorm1d(similarity_layers_sizes[0])
        if dropout:
            similarity_layers["dropout_0"] = nn.Dropout(dropout, inplace=True)
        prev_hidden_size = similarity_layers_sizes[0]
        for idx, hidden in enumerate(similarity_layers_sizes[1:], 1):
            similarity_layers["layer_{}".format(idx)] = nn.Linear(
                prev_hidden_size, hidden
            )
            similarity_layers["relu_{}".format(idx)] = nn.ReLU(inplace=True)
            similarity_layers["bn_{}".format(idx)] = nn.BatchNorm1d(hidden)
            if dropout:
                similarity_layers["dropout_{}".format(idx)] = nn.Dropout(
                    dropout, inplace=True
                )

        self.similarity = nn.Sequential(similarity_layers)

        self.output = nn.Linear(hidden, 1)

    def forward(self, image_1, image_2):
        left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)

        # for some weird reason, iv3 returns both
        # the 1000 class softmax AND the n_classes softmax
        # if train = True, so this is filthy, but necessary
        if self.training:
            left_features = left_features[0]
            right_features = right_features[0]

        features = torch.cat([left_features, right_features], 1)
        sim_features = self.similarity(features)
        output = self.output(sim_features)
        return output
