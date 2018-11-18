import os

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import dotdict, save_obj, load_obj

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
REV_DAMAGE_TYPES = {idx: lab for idx, lab in enumerate(DAMAGE_TYPES)}


class AIDataset(Dataset):

    def __init__(self,
                 directory,
                 name='train',
                 inputSize=(32, 32),
                 transforms=None):

        self.BEFORE_FOLDER = os.path.join(directory, 'Before')

        self.AFTER_FOLDER = os.path.join(directory, 'After')

        self.CACHED_DATA_FOLDER = os.path.join('.', 'cached')
        if not os.path.exists(self.CACHED_DATA_FOLDER):
            os.makedirs(self.CACHED_DATA_FOLDER)
        self.MAP_FILE = os.path.join('.', 'map.pkl')
        self.SPLIT_FILE = os.path.join('.', 'map-split.pkl')

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

        GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, geojson_file["train"])

        self.df = geopandas.read_file(GEOJSON_FILE)
        dataset_json = json.loads(self.df.to_json())
        features_json = dataset_json['features']

        self.inputSize = inputSize
        self.transforms = transforms

        if not os.path.isfile(self.MAP_FILE):
            self.createDatapoints(features_json)

        if not os.path.isfile(self.SPLIT_FILE):
            self.splitDatapoints()

        cached_mappings = load_obj(self.SPLIT_FILE)

        self.datapoints = cached_mappings[self.name]['file']
        self.indexes = cached_mappings[self.name]['index']

    def __len__(self):
        return len(self.datapoints)

    def __getitem__(self, idx):
        objectID = self.datapoints[idx]
        df_index = self.indexes[idx]
        before_image, after_image = self.loadDatapointImages(objectID)
        damage = self.df['_damage'][df_index]

        if self.transforms:
            before_image = self.transforms(before_image)
            after_image = self.transforms(after_image)

        return (before_image, after_image, self.onehot(damage))

    def createDatapoints(self, features):

        logger.info('Feature Size {}'.format(len(features)))

        BEFORE_FILE = os.path.join(self.BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

        feature_file_mapping = []
        feature_index_mapping = []

        with rasterio.open(BEFORE_FILE) as source_before_image:

            count = 0

            for index, feature in enumerate(features):

                # filter based on damage
                damage = feature['properties']['_damage']

                if damage not in DAMAGE_TYPES:
                    continue

                geometry = feature['geometry']

                # filter unstable data
                if geometry is None:
                    continue

                bounds = self.df['geometry'][index][0].bounds
                geoms = self.makesquare(*bounds)

                # identify data point
                objectID = feature['properties']['OBJECTID']

                try:
                    before_file = self.getCroppedImage(
                        source_before_image, geoms, 'b{}.png'.format(objectID))
                    after_file = self.getAfterImage(
                        geoms, 'a{}.png'.format(objectID))
                    if (before_file is not None) and os.path.isfile(before_file) and (after_file is not None) and os.path.isfile(after_file):
                        feature_file_mapping.append(objectID)
                        feature_index_mapping.append(index)
                        count += 1
                except ValueError as ve:
                    continue

        save_obj({
            'file': feature_file_mapping,
            'index': feature_index_mapping
        }, self.MAP_FILE)

        logger.info('Created {} Datapoints'.format(count))

        self.splitDatapoints()

    def splitDatapoints(self):
        cached_mappings = load_obj(self.MAP_FILE)

        datapoints = cached_mappings['file']
        indexes = cached_mappings['index']

        allIndexes = list(range(len(datapoints)))

        np.random.shuffle(allIndexes)

        training_offset = int(len(allIndexes) * 0.8)

        validation_offset = int(len(allIndexes) * 0.9)

        training_indexes = allIndexes[:training_offset]

        validation_indexes = allIndexes[training_offset:validation_offset]

        testing_indexes = allIndexes[validation_offset:]

        save_obj({
            'train': self.getValues(datapoints, indexes, training_indexes),
            'val': self.getValues(datapoints, indexes, validation_indexes),
            'test': self.getValues(datapoints, indexes, testing_indexes)
        }, self.SPLIT_FILE)


    def getValues(self, datapoints, indexes, selectIndexes):
        mapping = {
            'file': [datapoints[i] for i in selectIndexes],
            'index': [indexes[i] for i in selectIndexes]
        }
        return mapping


    def loadDatapointImages(self, objectID):
        before_image = Image.open(os.path.join(
            self.CACHED_DATA_FOLDER, 'b{}.png'.format(objectID)))
        after_image = Image.open(os.path.join(
            self.CACHED_DATA_FOLDER, 'a{}.png'.format(objectID)))
        return before_image, after_image

    def getCroppedImage(self, source, geometry, name):
        image, transform = rasterio.mask.mask(source, geometry, crop=True)
        out_meta = source.meta.copy()
        if np.sum(image) > 0:
            # save the resulting raster
            out_meta.update({
                "driver": "PNG",
                "height": image.shape[1],
                "width": image.shape[2],
                "transform": transform
            })
            file_path = os.path.join(self.CACHED_DATA_FOLDER, name)
            with rasterio.open(file_path, "w", **out_meta) as dest:
                dest.write(image)
            return file_path
        return None

    def getAfterImage(self, geometry, name):
        after_files = [os.path.join(self.AFTER_FOLDER, after_file)
                       for after_file in os.listdir(self.AFTER_FOLDER)]
        for index, file in enumerate(after_files):
            try:
                with rasterio.open(file) as after_file:
                    return self.getCroppedImage(after_file, geometry, name)
            except:
                pass
        return None

    def onehot(self, damage):
        index = DAMAGE_TYPES.index(damage)
        one_hot = [0] * len(DAMAGE_TYPES)
        one_hot[index] = 1
        return np.array(one_hot, dtype=np.long)

    def unonehot(self, onhot):
        # what a horrible name
        pass

    def makesquare(self, minx, miny, maxx, maxy):
        rangeX = maxx - minx
        rangeY = maxy - miny

        # 20 refers to 5% added to each side
        extension_factor = 20

        # Set image to a square if not square
        if rangeX == rangeY:
            pass
        elif rangeX > rangeY:
            difference_range = rangeX - rangeY
            miny -= difference_range/2
            maxy += difference_range/2
        elif rangeX < rangeY:
            difference_range = rangeY - rangeX
            minx -= difference_range/2
            maxx += difference_range/2
        else:
            pass

        # update ranges
        rangeX = maxx - minx
        rangeY = maxy - miny

        # add some extra border
        minx -= rangeX/extension_factor
        maxx += rangeX/extension_factor
        miny -= rangeY/extension_factor
        maxy += rangeY/extension_factor
        geoms = [{
            "type": "MultiPolygon",
            "coordinates": [[[
                [minx, miny],
                [minx, maxy],
                [maxx, maxy],
                [maxx, miny],
                [minx, miny]
            ]]]
        }]

        return geoms


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
