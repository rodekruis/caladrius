import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import dotdict

import rasterio
import geopandas
import json

import numpy as np
from PIL import Image

import rasterio.mask
import rasterio.features
import rasterio.warp

import logging

logger = logging.getLogger(__name__)

DAMAGE_TYPES = ['destroyed', 'significant', 'partial', 'none']


class AIDataset(Dataset):

    def __init__(self,
                 directory,
                 name='train',
                 inputSize=(32, 32),
                 transforms=None):

        self.BEFORE_FOLDER = os.path.join(directory, 'Before')

        self.AFTER_FOLDER = os.path.join(directory, 'After')

        GEOJSON_FOLDER = os.path.join(directory, 'Building Info')

        ALL_BUILDINGS_GEOJSON_FILE = os.path.join(
            GEOJSON_FOLDER, 'AllBuildingOutline.geojson')
        all_buildings_df = geopandas.read_file(ALL_BUILDINGS_GEOJSON_FILE)
        self.all_buildings_json = json.loads(all_buildings_df.to_json())

        geojson_file = {
            'train': 'TrainingDataset.geojson',
            'test_1': 'TestSet_1.geojson',
            'test_2': 'TestSet_2.geojson'
        }

        self.name = name

        GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, geojson_file[name])

        df = geopandas.read_file(GEOJSON_FILE)
        dataset_json = json.loads(df.to_json())
        features_json = dataset_json['features']

        self.inputSize = inputSize
        self.transforms = transforms

        self.datapoints = []

        # populate datapoints
        self.loadDatapoints(features_json)

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        return self.datapoints[idx]

    def loadDatapoints(self, features):

        logger.info('Raw Dataset Size {}'.format(len(features)))

        BEFORE_FILE = os.path.join(self.BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

        with rasterio.open(BEFORE_FILE) as source_before_image:

            for index, feature in enumerate(features):
                # initialize empty datapoint
                datapoint = dotdict({})

                datapoint.id = int(feature['id'])
                geometry = feature['geometry']

                # before image
                if geometry is None:
                    continue

                try:
                    before_image = self.getCroppedImage(
                        source_before_image, geometry)
                except ValueError as ve:
                    continue

                # after image

                after_image = self.getAfterImage(geometry)

                # get image according to geometry
                # apply mask using geometry

                if self.transforms is None:
                    datapoint.before = before_image
                    datapoint.after = before_image
                else:
                    datapoint.before = self.transforms(before_image)
                    datapoint.after = self.transforms(before_image)

                damage = feature['properties']['_damage']

                if damage not in DAMAGE_TYPES:
                    continue

                datapoint.label = self.onehot(damage)

                # add to datapoints
                self.datapoints.append(datapoint)

        logger.info('Processed Dataset Size {}'.format(len(self.datapoints)))

    def getCroppedImage(self, source, geometry):
        image, _ = rasterio.mask.mask(source, [geometry], crop=True)
        cropped_image = Image.fromarray(np.moveaxis(image.filled(), 0, -1))
        return cropped_image

    def getAfterImage(self, geometry):
        # logger.info(self.AFTER_FOLDER)
        image = None
        return image

    def onehot(self, damage):
        index = DAMAGE_TYPES.index(damage)
        one_hot = [0] * len(DAMAGE_TYPES)
        one_hot[index] = 1
        return one_hot


class Datasets(object):

    def __init__(self, args, transforms):
        self.args = args
        self.dataPath = args.dataPath
        self.batchSize = args.batchSize
        self.transforms = transforms

    def load(self, set_name):
        assert set_name in {"train", "val", "test"}
        dataset = AIDataset(self.dataPath, name=set_name,
                            transforms=self.transforms[set_name])
        dataLoader = DataLoader(
            dataset, batch_size=self.batchSize, shuffle=(set_name == 'train'))

        return dataset, dataLoader
