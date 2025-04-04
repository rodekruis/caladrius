from collections import OrderedDict

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms

from utils import create_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = create_logger(__name__)


def get_pretrained_vgg(output_size, freeze=False):
    """
    Get the pretrained vgg model, and change it for our use
    Args:
        output_size (int): Size of the output of the last layer

    Returns:
        model_conv: Model with Inception_v3 as base
    """

    if freeze:
        print("freeze not implemented for VGG")

    # fetch pretrained vgg16 model
    model_conv = torchvision.models.vgg16(pretrained=True)

    model_conv.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, output_size),
    )
    return model_conv


def get_pretrained_vgg_transforms(set_name, no_augment=False, augment_type="original"):
    """
    Compose a series of image transformations to be performed on the input data
    These augmentations are done per batch! So no extra data is generated, but the transformations for every epoch on the same images are different
    Args:
        set_name (str): the dataset you want the transformations for. Can be "train", "validation", "test", "inference"

    Returns:
        Composition of transformations for given set name
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 300
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


class VggSiameseNetwork(nn.Module):
    def __init__(
        self,
        output_size=512,
        similarity_layers_sizes=[512, 512],
        dropout=0.5,
        output_type="regression",
        n_classes=None,
        freeze=False,
    ):
        """
        Construct the Siamese network
        Args:
            output_size (int): output size of the Inception v3 model
            similarity_layers_sizes (list of ints): output sizes of each similarity layer
            dropout (float): amount of dropout, same for each layer
            n_classes (int): if output type is classification, this indicates the number of classes
        """
        super().__init__()
        self.left_network = get_pretrained_vgg(output_size, freeze)
        self.right_network = get_pretrained_vgg(output_size, freeze)
        # print("left",self.left_network.classifier[0].out_features)

        similarity_layers = OrderedDict()
        # fully connected layer where input is concatenated features of the two inception models
        similarity_layers["layer_0"] = nn.Linear(
            output_size * 2, similarity_layers_sizes[0]
        )
        similarity_layers["relu_0"] = nn.ReLU(inplace=True)
        similarity_layers["bn_0"] = nn.BatchNorm1d(similarity_layers_sizes[0])
        if dropout:
            similarity_layers["dropout_0"] = nn.Dropout(dropout, inplace=True)
        prev_hidden_size = similarity_layers_sizes[0]
        # make a hidden layer for each entry in similarity_layers_sizes
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
        if output_type == "regression":
            # final layer with one output which is the amount of damage from 0 to 1
            self.output = nn.Linear(hidden, 1)
        elif output_type == "classification":
            self.output = nn.Linear(hidden, n_classes)

    def forward(self, image_1, image_2):
        """
        Define the feedforward sequence
        Args:
            image_1: Image fed in to left network
            image_2: Image fed in to right network

        Returns:
            Predicted output
        """
        left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)

        features = torch.cat([left_features, right_features], 1)
        sim_features = self.similarity(features)
        output = self.output(sim_features)
        return output
