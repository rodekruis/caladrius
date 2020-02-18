import os
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler
import torch
import torch.utils.data
import torchvision

import imageio
import numpy as np


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self, dataset, indices=None, num_samples=None, callback_get_label=None
    ):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # print("label to count",label_to_count)
        # self.n_classes=len(label_to_count.keys())
        # print("number classes",self.n_classes)

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.load_datapoint(idx)[-1]
        # if isinstance(dataset, torchvision.datasets.MNIST):
        #     return dataset.train_labels[idx].item()
        # elif isinstance(dataset, torchvision.datasets.ImageFolder):
        #     return dataset.imgs[idx][1]
        # elif self.callback_get_label:
        #     return self.callback_get_label(dataset, idx)
        # else:
        #     raise NotImplementedError

    def __iter__(self):
        return (
            self.indices[i]
            for i in torch.multinomial(self.weights, self.num_samples, replacement=True)
        )

    def __len__(self):
        return self.num_samples


class CaladriusDataset(Dataset):
    def __init__(
        self,
        directory,
        set_name,
        labels_filename,
        transforms=None,
        max_data_points=None,
        augment_type="original",
    ):
        self.set_name = set_name
        self.directory = os.path.join(directory, set_name)
        self.labels_filename = labels_filename
        self.augment_type = augment_type
        if self.set_name == "inference":
            self.datapoints = [
                filename
                for filename in tqdm(os.listdir(os.path.join(self.directory, "before")))
            ]
        else:
            with open(
                os.path.join(self.directory, self.labels_filename)  # "labels.txt")
            ) as labels_file:
                self.datapoints = [x.strip() for x in tqdm(labels_file.readlines())]
        if max_data_points is not None:
            self.datapoints = self.datapoints[:max_data_points]
        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.load_datapoint(idx)

        if self.transforms:
            # datapoint[1] = self.transforms(imageio.imread(datapoint[1]))
            # datapoint[2] = self.transforms(imageio.imread(datapoint[2]))
            if self.augment_type == "equalization":
                datapoint[1] = np.array(datapoint[1])
                datapoint[2] = np.array(datapoint[2])
                datapoint[1] = self.transforms(image=datapoint[1])["image"].float()
                datapoint[2] = self.transforms(image=datapoint[2])["image"].float()
            else:
                datapoint[1] = self.transforms(datapoint[1])
                datapoint[2] = self.transforms(datapoint[2])

        return tuple(datapoint)

    def load_datapoint(self, idx):
        line = self.datapoints[idx]
        if self.set_name == "inference":
            filename = line
        else:
            filename, damage = line.split(" ")
        # before_image = imageio.imread(os.path.join(self.directory, "before", filename))
        # after_image = imageio.imread(os.path.join(self.directory, "after", filename))
        before_image = Image.open(os.path.join(self.directory, "before", filename))
        after_image = Image.open(os.path.join(self.directory, "after", filename))
        if self.set_name == "inference":
            datapoint = [filename, before_image, after_image]
        else:
            datapoint = [filename, before_image, after_image, float(damage)]
        return datapoint


class Datasets(object):
    def __init__(self, args, transforms):
        self.args = args
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.transforms = transforms
        self.number_of_workers = args.number_of_workers
        self.max_data_points = args.max_data_points
        self.label_file = args.label_file
        self.sample_data = args.sample_data
        self.augment_type = args.augment_type

    def load(self, set_name):
        assert set_name in {"train", "validation", "test", "inference"}
        dataset = CaladriusDataset(
            self.data_path,
            set_name,
            self.label_file,
            transforms=self.transforms[set_name],
            max_data_points=self.max_data_points,
            augment_type=self.augment_type,
        )

        if self.sample_data:
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                # shuffle=(set_name == "train"),
                num_workers=self.number_of_workers,
                drop_last=True,
                sampler=ImbalancedDatasetSampler(dataset)
                if (set_name == "train")
                else None,
            )
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=(set_name == "train"),
                num_workers=self.number_of_workers,
                drop_last=True,
                # sampler=ImbalancedDatasetSampler(dataset)
                # if (set_name == "train")
                # else None,
            )
            #
            # if set_name == "train":
            #     import matplotlib.pyplot as plt
            #     def show(data_loader):
            #         # print(next(iter(data_loader)))
            #         filenames, before_images, after_images, labels = next(iter(data_loader))
            #         # images = torch.stack([before_images,after_images],dim=0)
            #         from torchvision.utils import make_grid
            #         npimg = make_grid(after_images, normalize=True, pad_value=.5).numpy()
            #         import matplotlib.pyplot as plt
            #         fig, ax = plt.subplots(figsize=((13, 5)))
            #         import numpy as np
            #         ax.imshow(np.transpose(npimg, (1, 2, 0)))
            #         plt.setp(ax, xticks=[], yticks=[])
            #         plt.show()
            #         return fig, ax
            #     fig,ax=show(data_loader)
            #     plt.show()

        return dataset, data_loader
