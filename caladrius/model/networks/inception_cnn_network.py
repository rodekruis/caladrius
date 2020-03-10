from collections import OrderedDict

import torchvision
from torch import nn

from utils import create_logger

logger = create_logger(__name__)


def get_pretrained_iv3(output_size, freeze=False):
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
        if "Conv2d_4a_3x3" in ct and not freeze:
            for params in child.parameters():
                params.requires_grad = True
        ct.append(name)
    return model_conv


class InceptionCNNNetwork(nn.Module):
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
        Construct the CNN network
        Args:
            output_size (int): output size of the Inception v3 model
            similarity_layers_sizes (list of ints): output sizes of each similarity layer
            dropout (float): amount of dropout, same for each layer
            n_classes (int): if output type is classification, this indicates the number of classes
        """
        super().__init__()
        # self.left_network = get_pretrained_iv3(output_size, freeze)
        self.right_network = get_pretrained_iv3(output_size, freeze)

        similarity_layers = OrderedDict()
        # fully connected layer where input is concatenated features of the two inception models
        similarity_layers["layer_0"] = nn.Linear(
            output_size, similarity_layers_sizes[0]
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
        # left_features = self.left_network(image_1)
        right_features = self.right_network(image_2)

        # for some weird reason, iv3 returns both
        # the 1000 class softmax AND the n_classes softmax
        # if train = True, so this is filthy, but necessary
        if self.training:
            # left_features = left_features[0]
            right_features = right_features[0]

        features = right_features  # torch.cat([left_features, right_features], 1)
        sim_features = self.similarity(features)
        output = self.output(sim_features)
        return output
