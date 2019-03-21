import os
import sys

from shutil import move, copyfile

import rasterio
import geopandas
import json

import numpy as np
# from PIL import Image
from tqdm import tqdm

import rasterio.mask
import rasterio.features
import rasterio.warp

import logging

logger = logging.getLogger(__name__)
logging.getLogger('fiona').setLevel(logging.ERROR)
logging.getLogger('fiona.collection').setLevel(logging.ERROR)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)


def exceptionLogger(exceptionType, exceptionValue, exceptionTraceback):
    logger.error("Uncaught Exception", exc_info=(
        exceptionType, exceptionValue, exceptionTraceback))


sys.excepthook = exceptionLogger

# supported damage types
DAMAGE_TYPES = ['destroyed', 'significant', 'partial', 'none']

# input
ROOT_DIRECTORY = os.path.join('data', 'RC Challenge 1', '1')

BEFORE_FOLDER = os.path.join(ROOT_DIRECTORY, 'Before')
AFTER_FOLDER = os.path.join(ROOT_DIRECTORY, 'After')

GEOJSON_FOLDER = os.path.join(ROOT_DIRECTORY, 'Building Info')

ALL_BUILDINGS_GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'AllBuildingOutline.geojson')
GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'TrainingDataset.geojson')

# output
TARGET_DATA_FOLDER = os.path.join('data', 'Sint-Maarten-2017')
os.makedirs(TARGET_DATA_FOLDER, exist_ok=True)

# copy geojson files for visualization
coordinates_file = os.path.join(TARGET_DATA_FOLDER, 'coordinates.geojson')
copyfile(GEOJSON_FILE, coordinates_file)

# cache
TEMP_DATA_FOLDER = os.path.join(TARGET_DATA_FOLDER, 'temp')
os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

LABELS_FILE = os.path.join(TEMP_DATA_FOLDER, 'labels.txt')


def damage_quantifier(category):
    stats = {
        'none': {
            'mean': 0.2,
            'std': 0.2
        },
        'partial': {
            'mean': 0.55,
            'std': 0.15
        },
        'significant': {
            'mean': 0.85,
            'std': 0.15
        }
    }

    if category == 'none':
        value = np.random.normal(stats['none']['mean'], stats['none']['std'])
    elif category == 'partial':
        value = np.random.normal(stats['partial']['mean'], stats['partial']['std'])
    else:
        value = np.random.normal(stats['significant']['mean'], stats['significant']['std'])

    return np.clip(value, 0.0, 1.0)


def makesquare(minx, miny, maxx, maxy):
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


def getCroppedImage(source, geometry, folder, name):
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
        directory = os.path.join(TEMP_DATA_FOLDER, folder)
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, name)
        with rasterio.open(file_path, 'w', **out_meta) as dest:
            dest.write(image)
        return file_path
    return None


def getAfterImage(geometry, name):
    after_files = [os.path.join(AFTER_FOLDER, after_file)
                   for after_file in os.listdir(AFTER_FOLDER)]
    for index, file in enumerate(after_files):
        try:
            with rasterio.open(file) as after_file:
                return getCroppedImage(after_file, geometry, 'after', name)
        except:
            pass
    return None


def createDatapoints(features, df):

    logger.info('Feature Size {}'.format(len(features)))

    BEFORE_FILE = os.path.join(BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

    with open(LABELS_FILE, 'w+') as labels_file:
        with rasterio.open(BEFORE_FILE) as source_before_image:

            count = 0

            for index, feature in enumerate(tqdm(features)):

                # filter based on damage
                damage = feature['properties']['_damage']

                if damage not in DAMAGE_TYPES:
                    continue

                geometry = feature['geometry']

                # filter unstable data
                if geometry is None:
                    continue

                bounds = df['geometry'][index][0].bounds
                geoms = makesquare(*bounds)

                # identify data point
                objectID = feature['properties']['OBJECTID']

                try:
                    before_file = getCroppedImage(source_before_image, geoms, 'before', '{}.png'.format(objectID))
                    after_file = getAfterImage(geoms, '{}.png'.format(objectID))
                    if (before_file is not None) and os.path.isfile(before_file) and (after_file is not None) and os.path.isfile(after_file):
                        labels_file.write('{0}.png {1:.4f}\n'.format(objectID, damage_quantifier(damage)))
                        count += 1
                except ValueError as ve:
                    continue

    logger.info('Created {} Datapoints'.format(count))


def splitDatapoints(filepath):

    with open(filepath) as file:
        datapoints = file.readlines()

    allIndexes = list(range(len(datapoints)))

    np.random.shuffle(allIndexes)

    training_offset = int(len(allIndexes) * 0.8)

    validation_offset = int(len(allIndexes) * 0.9)

    training_indexes = allIndexes[:training_offset]

    validation_indexes = allIndexes[training_offset:validation_offset]

    testing_indexes = allIndexes[validation_offset:]

    split_mappings = {
        'train': [datapoints[i] for i in training_indexes],
        'validation': [datapoints[i] for i in validation_indexes],
        'test': [datapoints[i] for i in testing_indexes]
    }

    for split in split_mappings:

        split_filepath = os.path.join(TARGET_DATA_FOLDER, split)
        os.makedirs(split_filepath, exist_ok=True)

        split_labels_file = os.path.join(split_filepath, 'labels.txt')

        split_before_directory = os.path.join(split_filepath, 'before')
        os.makedirs(split_before_directory, exist_ok=True)

        split_after_directory = os.path.join(split_filepath, 'after')
        os.makedirs(split_after_directory, exist_ok=True)

        with open(split_labels_file, 'w+') as split_file:
            for datapoint in tqdm(split_mappings[split]):
                datapoint_name = datapoint.split(' ')[0]

                before_src = os.path.join(TEMP_DATA_FOLDER, 'before', datapoint_name)
                after_src = os.path.join(TEMP_DATA_FOLDER, 'after', datapoint_name)

                before_dst = os.path.join(split_before_directory, datapoint_name)
                after_dst = os.path.join(split_after_directory, datapoint_name)
                
                # print('{} => {} !! {}'.format(before_src, before_dst, os.path.isfile(before_src)))
                move(before_src, before_dst)

                # print('{} => {} !! {}'.format(after_src, after_dst, os.path.isfile(after_src)))
                move(after_src, after_dst)

                split_file.write(datapoint)

    return split_mappings


if __name__ == '__main__':

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join('.', 'run.log')),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    all_buildings_df = geopandas.read_file(ALL_BUILDINGS_GEOJSON_FILE)
    all_buildings_json = json.loads(all_buildings_df.to_json())

    df = geopandas.read_file(GEOJSON_FILE)
    dataset_json = json.loads(df.to_json())
    features_json = dataset_json['features']

    cached_mappings = createDatapoints(features_json, df)
    split_mappings = splitDatapoints(LABELS_FILE)
