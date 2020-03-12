import random
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torchvision
from torch import nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

from utils import create_logger


logger = create_logger(__name__)


def pair_transforms(before_image, after_image):

    if random.random() > 0.5:
        before_image = np.flip(before_image, 0)
        after_image = np.flip(after_image, 0)

    if random.random() > 0.5:
        before_image = np.flip(before_image, 1)
        after_image = np.flip(after_image, 1)

    if random.random() > 0.5:
        before_image = np.rot90(before_image, axes=(0, 1))
        after_image = np.rot90(after_image, axes=(0, 1))

    return np.array(before_image).copy(), np.array(after_image).copy()


def get_pretrained_iv3_transforms(set_name):
    """
    Compose a series of image transformations to be performed on the input data
    Args:
        set_name (str): the dataset you want the transformations for. Can be "train", "validation", "test", "inference"

    Returns:
        Composition of transformations for given set name
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    input_shape = 299

    resize_transform = transforms.Resize(scale)
    random_resized_crop_transform = transforms.RandomResizedCrop(input_shape)

    post_transforms = transforms.Compose(
        [
            # transforms.ToPILImage(),
            transforms.CenterCrop(input_shape),
            # converts image to type Torch and normalizes [0,1]
            transforms.ToTensor(),
            # normalizes [-1,1]
            transforms.Normalize(mean, std),
        ]
    )

    def tranform_function(before_image, after_image):
        # resize every image to scale x scale pixels
        before_image = resize_transform(before_image)
        after_image = resize_transform(after_image)

        if set_name == "train":
            # crop every image to input_shape x input_shape pixels
            i, j, h, w = random_resized_crop_transform.get_params(
                before_image,
                random_resized_crop_transform.scale,
                random_resized_crop_transform.ratio,
            )
            before_image = F.resized_crop(
                before_image,
                i,
                j,
                h,
                w,
                random_resized_crop_transform.size,
                random_resized_crop_transform.interpolation,
            )
            after_image = F.resized_crop(
                after_image,
                i,
                j,
                h,
                w,
                random_resized_crop_transform.size,
                random_resized_crop_transform.interpolation,
            )
            before_image, after_image = pair_transforms(before_image, after_image)
            before_image = Image.fromarray(before_image)
            after_image = Image.fromarray(after_image)
        # for testing and validation we don't want any permutations of the image, solely cropping and normalizing
        before_image = post_transforms(before_image)
        after_image = post_transforms(after_image)
        return before_image, after_image

    return tranform_function


def get_pretrained_iv3(output_size):
    """
    Get the pretrained Inception_v3 model, and change it for our use
    Args:
        output_size (int): Size of the output of the last layer

    Returns:
        model_conv: Model with Inception_v3 as base
    """
    # fetch pretrained inception_v3 model
    model_conv = torchvision.models.inception_v3(pretrained=True)

    # requires_grad indicates if parameter is learnable
    # so here set all parameters to non-learnable
    for i, param in model_conv.named_parameters():
        param.requires_grad = False

    # want to create own fully connected layer instead of using pretrained layer
    # get number of input features to fully connected layer
    num_ftrs = model_conv.fc.in_features
    # creaty fully connected layer
    model_conv.fc = nn.Linear(num_ftrs, output_size)

    # want almost all parameters learnable except for first few layers
    # so here set most parameters to learnable
    # idea is that first few layers learn types of features that are the same in all types of images --> don't have to retrain
    ct = []
    for name, child in model_conv.named_children():
        if "Conv2d_4a_3x3" in ct:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)
    return model_conv


class InceptionSiameseNetwork(nn.Module):
    def __init__(
        self,
        output_size=512,
        similarity_layers_sizes=[512, 512],
        dropout=0.5,
        output_type="regression",
        n_classes=None,
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
        self.left_network = get_pretrained_iv3(output_size)
        self.right_network = get_pretrained_iv3(output_size)

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
