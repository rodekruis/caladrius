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
        filename, before_image, after_image, damage = self.load_datapoint(idx)

        if self.transforms:
            before_image = self.transforms(before_image)
            after_image = self.transforms(after_image)

        return (filename, before_image, after_image, damage)

    def load_datapoint(self, idx):
        line = self.datapoints[idx]
        filename, damage = line.split(' ')
        before_image = Image.open(os.path.join(self.directory,
                                                'before',
                                                filename))
        after_image = Image.open(os.path.join(self.directory,
                                                'after',
                                                filename))
        return filename, before_image, after_image, float(damage)


class Datasets(object):

    def __init__(self, args, transforms):
        self.args = args
        self.data_path = args.data_path
        self.batch_size = args.batch_size
        self.transforms = transforms
        self.number_of_workers = args.number_of_workers
        self.max_data_points = args.max_data_points

    def load(self, set_name):
        assert set_name in {'train', 'validation', 'test'}
        dataset = CaladriusDataset(os.path.join(self.data_path, set_name),
                                    transforms=self.transforms[set_name],
                                    max_data_points=self.max_data_points)
        data_loader = DataLoader(dataset,
                                    batch_size=self.batch_size,
                                    shuffle=(set_name == 'train'),
                                    num_workers=self.number_of_workers,
                                    drop_last=True)

        return dataset, data_loader
