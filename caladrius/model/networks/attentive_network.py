from collections import OrderedDict

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from utils import create_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = create_logger(__name__)


def get_pretrained_eff4(output_size, freeze=False):
    model_conv = torchvision.models.efficientnet_b4(pretrained=True)
    num_ftrs = model_conv.classifier[1].in_features
    x = model_conv.classifier[1]
    model_conv.classifier[1] = nn.Linear(num_ftrs, output_size)
    del x
    return model_conv

def get_pretrained_attentive_transforms(set_name, no_augment=False, augment_type="original"):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    scale = 256
    input_shape = 224

    if no_augment:
        train_transform = transforms.Compose(
            [
                transforms.Resize((input_shape, input_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize((input_shape, input_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    elif augment_type == "original":
        train_transform = transforms.Compose(
            [
                # resize every image to scale x scale pixels
                transforms.Resize(scale),
                # crop every image to input_shape x input_shape pixels.
                # This is needed for the inception model.
                # we first scale and then crop to have translation variation, i.e. buildings is not always in the centre.
                # In this way model is less sensitive to translation variation in the test set.
                transforms.RandomResizedCrop(input_shape),
                # flips image horizontally with a probability of 0.5 (i.e. half of images are flipped)
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                # rotates image randomly between -90 and 90 degrees
                transforms.RandomRotation(degrees=90),
                # converts image to type Torch and normalizes [0,1]
                transforms.ToTensor(),
                # normalizes [-1,1]
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                # for testing and validation we don't want any permutations of the image, solely cropping and normalizing
                transforms.Resize(scale),
                transforms.CenterCrop(input_shape),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    elif augment_type == "paper":
        train_transform = transforms.Compose(
            [
                transforms.Resize(input_shape),
                # # accidentally added rotation twice, one of the tests was run with this
                # transforms.RandomRotation(degrees=40),
                transforms.RandomAffine(degrees=40, translate=(0.2, 0.2), shear=11.5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(input_shape, scale=(0.8, 1)),
                # converts image to type Torch and normalizes [0,1]
                transforms.ToTensor(),
                # normalizes [-1,1]
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                # for testing and validation we don't want any permutations of the image, solely cropping and normalizing
                transforms.Resize((input_shape, input_shape)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    elif augment_type == "equalization":
        train_transform = A.Compose(
            [
                A.Resize(scale, scale),
                A.RandomResizedCrop(input_shape, input_shape),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomRotate90(),
                A.CLAHE(p=1),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

        test_transform = A.Compose(
            [
                A.Resize(scale, scale),
                A.CenterCrop(input_shape, input_shape),
                A.CLAHE(p=1),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]
        )

    return {
        "train": train_transform,
        "validation": test_transform,
        "test": test_transform,
        "inference": test_transform,
    }[set_name]


class AttentiveNetwork(nn.Module):
    def __init__(
        self,
        output_size=512,
        similarity_layers_sizes=[512, 512],
        dropout=0.5,
        output_type="regression",
        n_classes=None,
        freeze=False,
    ):
        super().__init__()
        self.left_network = get_pretrained_eff4(output_size, freeze)
        self.right_network = get_pretrained_eff4(output_size, freeze)

        self.left_layers = list(self.left_network.features.children()) +\
            [self.left_network.avgpool, self.left_network.classifier]
        self.right_layers = list(self.right_network.features.children()) +\
            [self.right_network.avgpool, self.right_network.classifier]

        output_channels = [24, 32, 56, 112, 160, 272, 448, 1792]

        for i in range(1, len(self.left_layers) - 2):
            attention_layer = nn.Conv2d(in_channels=output_channels[i-1],
                                        out_channels=output_channels[i-1],
                                        kernel_size=1,
                                        groups=1)
            setattr(self, 'attention%d_1' % i, attention_layer)

        if dropout:
            self.drop = nn.Dropout(dropout, inplace=False)

        if output_type == "regression":
            self.output = nn.Linear(hidden, 1)
        elif output_type == "classification":
            self.output = nn.Linear(output_size, n_classes)

        self.relu = nn.ReLU()

    def forward(self, image_1, image_2):
        left_features = image_1
        right_features = image_2

        for i in range(len(self.left_layers)):
            left_features = self.left_layers[i](left_features)
            if i < len(self.left_layers) - 2:
                right_features = self.right_layers[i](right_features)

            if 0 < i < len(self.left_layers) - 2:
                attention_1 = getattr(self, 'attention%d_1' % i)

                attention_1 = torch.sigmoid(attention_1(left_features))

                left_features = left_features * attention_1 + right_features * (1 - attention_1)
            elif i == len(self.left_layers) - 2:
                left_features = left_features[..., 0, 0]

        left_features = self.relu(left_features)
        left_features = self.drop(left_features)       

        output = self.output(left_features)
        return output