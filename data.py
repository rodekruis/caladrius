import os

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from PIL import Image

import logging

logger = logging.getLogger(__name__)


class CaladriusDataset(Dataset):

    def __init__(self,
                 directory,
                 split='train',
                 inputSize=(32, 32),
                 transforms=None):
        self.directory = directory
        with open(os.path.join(directory, 'labels.txt')) as labels_file:
            self.datapoints = [x.strip() for x in tqdm(labels_file.readlines())]
        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        before_image, after_image, damage = self.loadDatapoint(idx)

        if self.transforms:
            before_image = self.transforms(before_image)
            after_image = self.transforms(after_image)

        return (before_image, after_image, damage)

    def loadDatapoint(self, idx):
        line = self.datapoints[idx]
        filename, damage = line.split(' ')
        before_image = Image.open(os.path.join(self.directory, 'before', filename))
        after_image = Image.open(os.path.join(self.directory, 'after', filename))
        return before_image, after_image, float(damage)


class Datasets(object):

    def __init__(self, args, transforms):
        self.args = args
        self.dataPath = args.dataPath
        self.batchSize = args.batchSize
        self.transforms = transforms

    def load(self, set_name):
        assert set_name in {'train', 'validation', 'test'}
        dataset = CaladriusDataset(os.path.join(self.dataPath, set_name), transforms=self.transforms[set_name])
        dataLoader = DataLoader(dataset, batch_size=self.batchSize, shuffle=(set_name == 'train'), num_workers=8)

        return dataset, dataLoader
