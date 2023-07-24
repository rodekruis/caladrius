import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from time import sleep
from torch.utils.data import Dataset, DataLoader
import torch
import torch.utils.data


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements such that all classes are equally represented
    Adjusted from https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
    Arguments:
        dataset: the dataset from which to sample
    """

    def __init__(self, dataset):

        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset)))

        # keep resampled dataset size the same as original
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [
            1.0 / label_to_count[self._get_label(dataset, idx)] for idx in self.indices
        ]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.load_datapoint(idx)[-1]

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
        active_idx,
        labels_filename,
        transforms=None,
        max_data_points=None,
        augment_type="original",
    ):
        self.set_name = set_name
        #self.directory = "./test_small/"
        #os.chdir("pipplegeneral1/xview_set/")
        self.directory = os.path.join(directory, set_name) #"./test_small"
        self.labels_filename = labels_filename
        self.augment_type = augment_type
        if self.set_name == "inference":
            self.datapoints = [
                filename
                for filename in tqdm(os.listdir(os.path.join(self.directory, "before")))
            ]
        else:
            #print(os.getcwd())
            with open(
                os.path.join(self.directory, self.labels_filename) # "labels.txt")
            ) as labels_file:
                self.datapoints = [x.strip() for x in tqdm(labels_file.readlines())]
        if max_data_points is not None:
            self.datapoints = self.datapoints[:max_data_points]
        if type(active_idx) == np.ndarray and self.set_name == 'train': #If active, only select the labelled images to train on. Todo: In practical situation, these still need to be labelled and added to a file, a framework for this is needed
            datapoints_numpy = np.array(self.datapoints)
            datapoints_numpy = datapoints_numpy[active_idx]
            self.datapoints = np.ndarray.tolist(datapoints_numpy)

        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        datapoint = self.load_datapoint(idx)

        if self.transforms:
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
        try:
            before_image = Image.open(os.path.join(self.directory, "before", filename))
            after_image = Image.open(os.path.join(self.directory, "after", filename))
        except FileNotFoundError:
            sleep(1)
            try:
                before_image = Image.open(os.path.join(self.directory, "before", filename))
                after_image = Image.open(os.path.join(self.directory, "after", filename))
            except FileNotFoundError:
                self.load_datapoint(idx-1)

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

    def load(self, set_name, active_idx = False, active_inference = False, tsne = False):
        assert set_name in {"train", "validation", "test", "inference"}
        dataset = CaladriusDataset(
            self.data_path,
            set_name,
            active_idx,
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
            if active_inference:
                data_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle = False,
                    num_workers=self.number_of_workers,
                )
            else:
                data_loader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=(set_name == "train"),
                    num_workers=self.number_of_workers,
                    drop_last=True,
                )

            # function to plot some examples of the data augmentation. Used for testing and research purposes
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
