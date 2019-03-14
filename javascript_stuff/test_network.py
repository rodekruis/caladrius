import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
import logging

logging.basicConfig(level=logging.INFO)

def get_pretrained_iv3(output_size):
    model_conv = torchvision.models.inception_v3(pretrained=True)

    for i, param in model_conv.named_parameters():
        param.requires_grad = False

    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, output_size)

    ct = []
    for name, child in model_conv.named_children():
        if 'Conv2d_4a_3x3' in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)

    return model_conv

inception_network = get_pretrained_iv3(512)
# print(inception_network)


# You must have images of size 299 x 299
# Batch size may NEVER be 1, always greater than 1
image = torch.rand(5, 3, 299, 299)
# print(image.shape)

x = inception_network(image)
