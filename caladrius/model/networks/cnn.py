import random

import numpy as np

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


def get_cnn_transforms(set_name):
    """
    Compose a series of image transformations to be performed on the input data
    Args:
        set_name (str): the dataset you want the transformations for. Can be "train", "validation", "test", "inference"

    Returns:
        Composition of transformations for given set name
    """
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 70
    input_shape = 64

    resize_transform = transforms.Resize(scale)
    random_resized_crop_transform = transforms.RandomResizedCrop(input_shape)

    post_transforms = transforms.Compose(
        [
            transforms.ToPILImage(),
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

        # for testing and validation we don't want any permutations of the image, solely cropping and normalizing
        before_image = post_transforms(before_image)
        after_image = post_transforms(after_image)
        return before_image, after_image

    return tranform_function


class CNN(nn.Module):
    def __init__(self, output_size):
        super().__init__()

        ## define the layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv_1_bn = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv_2_bn = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv_3_bn = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(4096, 1024)
        self.linear2 = nn.Linear(1024, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):  # B x 3 x 64 x 64
        x = self.pool(self.relu(self.conv_1_bn(self.conv1(x))))  # B x 16 x 32 x 32
        x = self.pool(self.relu(self.conv_2_bn(self.conv2(x))))  # B x 32 x 16 x 16
        x = self.pool(self.relu(self.conv_3_bn(self.conv3(x))))  # B x 64 x  8 x  8
        x = x.view(-1, 4096)  # B x 4096
        x = self.relu(self.linear1(x))  # B x 1024
        x = self.linear2(x)  # B x output_size
        return x
