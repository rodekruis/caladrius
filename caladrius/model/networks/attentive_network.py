import numpy as np
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from utils import create_logger
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from model.networks.axial_attention import AxialAttention


logger = create_logger(__name__)


def get_pretrained_eff0(output_size, freeze=False):
    model_conv = torchvision.models.efficientnet_b0(pretrained=True)
    num_ftrs = model_conv.classifier[-1].in_features
    x = model_conv.classifier[-1]
    model_conv.classifier[-1] = nn.Linear(num_ftrs, output_size)
    del x
    return model_conv


class CenterCropResize(object):
    def __init__(self, target_size):
        """ Center crop preserving aspect ratio."""
        self.target_size = target_size

    def __call__(self, img: Image.Image) -> Image.Image:
        center_crop = transforms.CenterCrop(min(img.width, img.height))
        img = center_crop(img)
        img = img.resize((self.target_size, self.target_size), Image.BICUBIC)
        return img


class RandomScalingAndCrop(object):
    def __init__(self, min_size, target_size):
        """ Samples random crop from an image with minimum min_size and then resizes it to target_size.
            Preserves aspect ratio.
            Smaller images are sampled more frequently."""
        self.min_size = min_size
        self.target_size = target_size
        self.center_crop_resize = CenterCropResize(self.target_size)

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            if img.width < self.min_size:
                raise ValueError(f"Image is too small to be cropped, {img.width}")

            prob = np.arange(1, img.width - self.min_size + 2, dtype=np.float32)[::-1] ** 2
            if len(prob) == 0:
                raise ValueError("Image is too small to be cropped")
            prob /= np.sum(prob)

            if img.height < self.min_size:
                raise ValueError(f"Image is too small to be cropped, {img.height}")

            prob2 = np.arange(1, img.height - self.min_size + 2, dtype=np.float32)[::-1] ** 2
            if len(prob2) == 0:
                raise ValueError("Image is too small to be cropped")

            prob2 /= np.sum(prob2)

            width = np.random.choice(np.arange(self.min_size, img.width  + 1, dtype=np.int32), size=1, p=prob)[0]
            init_x = np.random.randint(0, img.width - width+1)

            height = np.random.choice(np.arange(self.min_size, img.height  + 1, dtype=np.int32), size=1, p=prob2)[0]
            init_y = np.random.randint(0, img.height - height+1)

            img = img.crop((init_x, init_y, init_x + width, init_y + height))

            max_size = max(width, height)

            padding = (max_size - width, max_size - height)
            left_padding = padding[0] // 2
            top_padding = padding[1] // 2
            right_padding = padding[0] - left_padding
            bottom_padding = padding[1] - top_padding

            # Pad the image
            img = TF.pad(img, (left_padding, top_padding, right_padding, bottom_padding), padding_mode='reflect')

            img = img.resize((self.target_size, self.target_size), Image.BICUBIC)
        except ValueError as e:
            img = self.center_crop_resize(img)

        return img


def get_pretrained_attentive_transforms(set_name, no_augment=False, augment_type="original"):
    # https://pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0.html#torchvision.models.efficientnet_b0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # EffNetB0
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
                RandomScalingAndCrop(32, input_shape),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                torchvision.transforms.Pad(input_shape, padding_mode="reflect"),
                # rotates image randomly between -90 and 90 degrees
                transforms.RandomRotation(degrees=90, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
                torchvision.transforms.CenterCrop(input_shape),

                # converts image to type Torch and normalizes [0,1]
                transforms.ToTensor(),
                # normalizes [-1,1]
                transforms.Normalize(mean, std),
            ]
        )

        test_transform = transforms.Compose(
            [
                # for testing and validation we don't want any permutations of the image, solely cropping and normalizing
                CenterCropResize(input_shape),
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


def replace_stochastic_depth_with_identity(model):
    """ Replace all instances of torchvision.ops.StochasticDepth in the model with torch.nn.Identity.

    Parameters:
    - model: The PyTorch model to modify.

    Returns:
    - The modified model with Identity layers instead of StochasticDepth.
    """
    for name, module in model.named_children():
        if isinstance(module, torchvision.ops.StochasticDepth):
            setattr(model, name, torch.nn.Identity())
        else:
            replace_stochastic_depth_with_identity(module)
    return model


class AttentiveNetwork(nn.Module):
    def __init__(
        self,
        output_size=512,
        dropout=0.5,
        output_type="regression",
        n_classes=None,
        freeze=False,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.left_network = get_pretrained_eff0(output_size, freeze)

        self.right_network = get_pretrained_eff0(output_size, freeze)

        for i in range(len(list(self.left_network.features.children()))-1, 3, -1):
            del self.left_network.features[i]

        self.left_layers = list(self.left_network.features.children()) +\
            [self.left_network.avgpool, self.left_network.classifier]
        self.right_layers = list(self.right_network.features.children()) +\
            [self.right_network.avgpool, self.right_network.classifier]
        self.attention_layers = []

        def get_out_channels(layers):
            try:
                n_layers = len(layers)
            except TypeError:
                return 0

            for i in range(n_layers-1, -1, -1):
                try:
                    return layers[i].out_channels
                except AttributeError:
                    continue
            return 0

        output_channels = [get_out_channels(l) for l in self.right_layers]
        output_channels = output_channels[1:]

        # output_channels = [24, 32, 56, 112, 160, 272, 448, 1792, 0, 0]

        self.threshold = output_channels[2]

        self.right_layers[1] = replace_stochastic_depth_with_identity(self.right_layers[1])
        self.left_layers[1] = replace_stochastic_depth_with_identity(self.left_layers[1])

        # The first layer block of EfficientNet is shallow, so we start attending after the second layer
        for i in range(2, len(self.right_layers) - 2):
            if output_channels[i-1] <= self.threshold:
                self.right_layers[i] = replace_stochastic_depth_with_identity(self.right_layers[i])
                self.left_layers[i] = replace_stochastic_depth_with_identity(self.left_layers[i])

                attention_layer = AxialAttention(
                    dim = output_channels[i-1],
                    dim_index = 1,
                    dim_heads = 16,
                    heads = 1,
                    num_dimensions = 2,
                    sum_axial_out = True,
                    mask_padding=2 if i == 2 else 1
                )

                self.add_module('attention%d_1' % i, attention_layer)
                self.attention_layers.append(attention_layer)

        if dropout:
            self.drop = nn.Dropout(dropout, inplace=False)

        if output_type == "regression":
            self.output = nn.Linear(output_size, 1)

        elif output_type == "classification":
            self.output = nn.Linear(output_channels[-3], n_classes)

    def forward(self, image_1, image_2):
        left_features = image_1 # original
        right_features = image_2 # damaged

        for i in range(len(self.right_layers)-1):
            right_features = self.right_layers[i](right_features)

            if (i < len(self.right_layers) - 1) and (right_features.shape[1] <= self.threshold):
                left_features = self.left_layers[i](left_features)

            # Skipping the first block
            if 1 < i < len(self.right_layers) - 2:
                if right_features.shape[1] <= self.threshold:
                    attention_1 = self.get_submodule('attention%d_1' % i)

                    right_features = attention_1(left_features, right_features)

                    if i == 3:
                        right_features = right_features * left_features \
                            / torch.sqrt(torch.sum(right_features ** 2, 1, keepdims=True) * torch.sum(left_features ** 2, 1, keepdims=True) + 0.000001)

            elif i == len(self.right_layers) - 2:
                assert (right_features.shape[-1] == 1) and (right_features.shape[-2] == 1), right_features.shape

                right_features = torch.squeeze(right_features)

        right_features = self.drop(right_features)

        output = self.output(right_features)

        return output
