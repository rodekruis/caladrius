from collections import OrderedDict

import torch
from torch import nn

from model.networks.cnn import CNN, get_cnn_transforms

from utils import create_logger


logger = create_logger(__name__)


def get_light_siamese_transforms(*args):
    return get_cnn_transforms(*args)


def get_cnn(output_size):
    """
    Get a light CNN model
    Args:
        output_size (int): Size of the output of the last layer

    Returns:
        model_conv: CNN model
    """
    model_conv = CNN(output_size)

    for i, param in model_conv.named_parameters():
        param.requires_grad = True

    return model_conv


class LightSiameseNetwork(nn.Module):
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
            output_size (int): output size of the CNN model
            similarity_layers_sizes (list of ints): output sizes of each similarity layer
            dropout (float): amount of dropout, same for each layer
            n_classes (int): if output type is classification, this indicates the number of classes
        """
        super().__init__()
        self.left_network = get_cnn(output_size)
        self.right_network = get_cnn(output_size)

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
        if output_type == "regression":
            self.output = nn.Linear(hidden, 1)
        elif output_type == "classification":
            self.output = self.output = nn.Linear(hidden, n_classes)

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
