import os
from PIL import Image
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader


class CaladriusDataset(Dataset):

    def __init__(self,
                 directory,
                 transforms=None,
                 max_data_points=None):
        self.directory = directory
        with open(os.path.join(directory, 'labels.txt')) as labels_file:
            self.datapoints = [x.strip() for x in tqdm(labels_file.readlines())]
        if max_data_points is not None:
            self.datapoints = self.datapoints[:max_data_points]
        self.transforms = transforms

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        filename, before_image, after_image, damage = self.loadDatapoint(idx)

        if self.transforms:
            before_image = self.transforms(before_image)
            after_image = self.transforms(after_image)

        return (filename, before_image, after_image, damage)

    def loadDatapoint(self, idx):
        line = self.datapoints[idx]
        filename, damage = line.split(' ')
        before_image = Image.open(os.path.join(self.directory, 'before', filename))
        after_image = Image.open(os.path.join(self.directory, 'after', filename))
        return filename, before_image, after_image, float(damage)


class Datasets(object):

    def __init__(self, args, transforms):
        self.args = args
        self.dataPath = args.dataPath
        self.batchSize = args.batchSize
        self.transforms = transforms
        self.numberOfWorkers = args.numberOfWorkers
        self.maxDataPoints = args.maxDataPoints

    def load(self, set_name):
        assert set_name in {'train', 'validation', 'test'}
        dataset = CaladriusDataset(os.path.join(self.dataPath, set_name), transforms=self.transforms[set_name],
                                   max_data_points=self.maxDataPoints)
        dataLoader = DataLoader(dataset, batch_size=self.batchSize, shuffle=(set_name == 'train'), num_workers=self.numberOfWorkers)

        return dataset, dataLoader
