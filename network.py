import logging
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torch.nn.modules import loss as nnloss
from torch.optim import Adam

import torchvision.transforms as transforms

# TODO: DataParallel


log = logging.getLogger(__name__)


def get_pretrained_iv3(output_size, num_to_freeze=7):
    model_conv = torchvision.models.inception_v3(pretrained='imagenet')

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
            log.debug("{}, {}".format(name_2, params.requires_grad))

    return model_conv


def get_pretrained_iv3_transforms(set_name):
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    scale = 360
    input_shape = 299
    train_transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.RandomResizedCrop(input_shape),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    test_transform = transforms.Compose([
        transforms.Resize(scale),
        transforms.CenterCrop(input_shape),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    return {
        "train": train_transform,
        "val": test_transform,
        "test": test_transform
    }[set_name]


class SiameseNetwork(nn.Module):
    def __init__(self, n_classes, output_size=512, n_freeze=7,
                 similarity_layers_sizes=[512, 512], dropout=0.5):
        super().__init__()
        self.left_network = get_pretrained_iv3(
            output_size, num_to_freeze=n_freeze)
        self.right_network = get_pretrained_iv3(
            output_size, num_to_freeze=n_freeze)

        similarity_layers = OrderedDict()
        similarity_layers["layer_0"] = nn.Linear(
            output_size*2, similarity_layers_sizes[0])
        similarity_layers["relu_0"] = nn.ReLU(inplace=True)
        similarity_layers["bn_0"] = nn.BatchNorm1d(similarity_layers_sizes[0])
        if dropout:
            similarity_layers["dropout_0"] = nn.Dropout(
                dropout, inplace=True)
        prev_hidden_size = similarity_layers_sizes[0]
        for idx, hidden in enumerate(similarity_layers_sizes[1:], 1):
            similarity_layers["layer_{}".format(idx)] = nn.Linear(
                prev_hidden_size, hidden)
            similarity_layers["relu_{}".format(idx)] = nn.ReLU(inplace=True)
            similarity_layers["bn_{}".format(idx)] = nn.BatchNorm1d(hidden)
            if dropout:
                similarity_layers["dropout_{}".format(idx)] = nn.Dropout(
                    dropout, inplace=True)

        self.output = nn.Linear(hidden, n_classes)

        self.similarity = nn.Sequential(similarity_layers)

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


class QuasiSiameseNetwork(object):

    def __init__(self, config, n_freeze=7):
        assert config in ("soft-targets", "softmax")
        if config == "soft-targets":
            self.n_classes = 1
            self.criterion = nnloss.BCEWithLogitsLoss()
        else:
            # TODO: weights
            self.n_classes = 4
            self.criterion = nnloss.CrossEntropyLoss()

        self.model = SiameseNetwork(self.n_classes, n_freeze=n_freeze)
        self.transforms = {}
        for s in ("train", "val", "test"):
            self.transforms[s] = get_pretrained_iv3_transforms(s)

        log.debug("Num params: {}".format(
            len([_ for _ in self.model.parameters()])))
        self.optimizer = Adam(self.model.parameters())

    def run_epoch(self, epoch, loader, device, phase="train"):
        assert phase in ("train", "val", "test")

        self.model = self.model.to(device)

        log.info("Phase: {}, Epoch: {}".format(phase, epoch))

        if phase == 'train':
            self.model.train()  # Set model to training mode
        else:
            self.model.eval()

        running_loss = 0.0
        running_corrects = 0
        running_n = 0.0

        for idx, (image1, image2, labels) in enumerate(loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.to(device)

            if phase == "train":
                # zero the parameter gradients
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(image1, image2)
                _, preds = torch.max(outputs, 1)
                _, labels = torch.max(labels, 1)
                loss = self.criterion(preds, labels)

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * image1.size(0)
            running_corrects += torch.sum(preds == labels.data)
            running_n += image1.size(0)

            if batch_idx % 10 == 0:
                log.info("\tBatch {}: Loss: {:.4f} Acc: {:.4f}".format(
                    idx, running_loss / running_n, running_corrects.double() / running_n))

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / \
            self.dataset_sizes[phase]

        log.info('{}: Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))

        return epoch_loss, epoch_acc

    def train(self, n_epochs, datasets, device):
        train_set, train_loader = datasets.load("train")
        #val_set, val_loader = datasets.load("val")

        for epoch in range(n_epochs):
            # train network
            train_loss, train_acc = self.run_epoch(
                epoch, train_loader, device, phase="train")

            # eval on validation
            #self.run_epoch(epoch, loader, device, phase="val")

    def test(self, datasets, device):
        pass


if __name__ == '__main__':
    qsn = QuasiSiameseNetwork("soft-targets")
    print(qsn)
