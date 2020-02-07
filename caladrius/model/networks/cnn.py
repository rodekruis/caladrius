from torch import nn
import torchvision.transforms as transforms

from utils import create_logger


logger = create_logger(__name__)


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
        "inference": test_transform,
    }[set_name]


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
