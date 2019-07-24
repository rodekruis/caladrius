import os
import sys
import argparse

from shutil import move

import rasterio
import pandas as pd
import geopandas
from geopandas.tools import reverse_geocode

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

# Fraction of image pixels that must be non-zero
NONZERO_PIXEL_THRESHOLD = 0.90

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

# cache
TEMP_DATA_FOLDER = os.path.join(TARGET_DATA_FOLDER, 'temp')
os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

LABELS_FILE = os.path.join(TEMP_DATA_FOLDER, 'labels.txt')
ADDRESS_CACHE = os.path.join(TARGET_DATA_FOLDER, 'address_cache.esri')

# Administrative boundaries file
ADMIN_REGIONS_FILE = os.path.join(GEOJSON_FOLDER, 'admin_regions', 'sxm_admbnda_adm1.shp')


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


def saveImage(image, transform, out_meta, folder, name):
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


def getBeforeImage(source, geometry, name):
    image, transform = rasterio.mask.mask(source, geometry, crop=True)
    out_meta = source.meta.copy()
    good_pixel_frac = np.count_nonzero(image) / image.size
    if np.sum(image) > 0 and good_pixel_frac > NONZERO_PIXEL_THRESHOLD:
        return saveImage(image, transform, out_meta, 'before', name)
    return None


def getAfterImage(geometry, name):
    after_files = [os.path.join(AFTER_FOLDER, after_file)
                   for after_file in os.listdir(AFTER_FOLDER) if after_file.endswith('.tif')]
    image_list = []
    for index, file in enumerate(after_files):
        try:
            with rasterio.open(file) as after_file:
                image, transform = rasterio.mask.mask(after_file, geometry, crop=True)
                good_pixel_frac = np.count_nonzero(image) / image.size
                if np.sum(image) > 0 and good_pixel_frac > NONZERO_PIXEL_THRESHOLD:
                    image_list.append({'after_file': after_file,
                                       'good_pixel_frac': good_pixel_frac,
                                       'image': image,
                                       'transform': transform})
        except ValueError:
            pass
    if len(image_list) == 0:
        return None
    elif len(image_list) == 1:
        after_image = image_list[0]
    else:
        after_image = image_list[np.argmax(np.array([image['good_pixel_frac']
                                                     for image in image_list]))]
    return saveImage(after_image['image'], after_image['transform'],
                     after_image['after_file'].meta.copy(), 'after', name)


def createDatapoints(df):

    logger.info('Feature Size {}'.format(len(df)))

    BEFORE_FILE = os.path.join(BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

    with open(LABELS_FILE, 'w+') as labels_file:
        with rasterio.open(BEFORE_FILE) as source_before_image:

            count = 0

            for index, row in tqdm(df.iterrows(), total=df.shape[0]):

                # filter based on damage
                damage = row['_damage']
                if damage not in DAMAGE_TYPES:
                    continue

                bounds = row['geometry'].bounds
                geoms = makesquare(*bounds)

                # identify data point
                objectID = row['OBJECTID']

                before_file = getBeforeImage(source_before_image, geoms,'{}.png'.format(objectID))
                after_file = getAfterImage(geoms, '{}.png'.format(objectID))
                if (before_file is not None) and os.path.isfile(before_file) and (after_file is not None) \
                        and os.path.isfile(after_file):
                    labels_file.write('{0}.png {1:.4f}\n'.format(objectID, damage_quantifier(damage)))
                    count += 1

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


def query_address_api(df, address_api='openmapquest', address_api_key=None):

    logger.info("Querying address API")

    # Create the address data frame and cache file if it doesn't exist already
    if not os.path.exists(ADDRESS_CACHE):
        logger.info("Converting {} geometries to EPSG 4326, this meay take awhile ".format(len(df)))
        df_address = geopandas.GeoDataFrame(df.to_crs(epsg='4326').geometry, crs='epsg:4326')
        df_address['address'] = None
        logger.info("Creating new address cache file {}".format(ADDRESS_CACHE))
        df_address.to_file(ADDRESS_CACHE, driver='ESRI Shapefile')
    else:
        logger.info("Reading in previous address cache file {}".format(ADDRESS_CACHE))
        df_address = geopandas.read_file(ADDRESS_CACHE)

    empty_address = df_address.loc[pd.isna(df_address['address'])]
    logger.info("Querying for {} addresses".format(len(empty_address)))
    for row in tqdm(empty_address.itertuples(), total=empty_address.shape[0]):
        try:
            address = reverse_geocode(row.geometry.centroid,
                                      user_agent='caladrius',
                                      provider=address_api,
                                      api_key=address_api_key)
            df_address.loc[row.Index, 'address'] = address['address'][0]
            df_address.to_file(ADDRESS_CACHE, driver='ESRI Shapefile')
        except Exception as e:
            logger.exception('Geocoding failed for {latlon}'.format(latlon=row.geometry.centroid))
            continue


def create_geojson_for_visualization(df):

    logger.info("Adding boundary information for report")

    # Use centroids for the intersection, to avoid duplicates
    df['centroid'] = df.centroid
    df['shape'] = df['geometry']

    # Read in the admin regions
    admin_regions = geopandas.read_file(ADMIN_REGIONS_FILE).to_crs(df.crs)

    # Get the centroid intersection with the admin regions
    df.set_geometry('centroid', inplace=True, drop=True)
    df = geopandas.sjoin(df, admin_regions, how='left')
    df.set_geometry('shape', inplace=True, drop=True)

    # Add the addresses
    if os.path.exists(ADDRESS_CACHE):
        logger.info("Adding address information for report")
        df_address = geopandas.read_file(ADDRESS_CACHE)
        df['address'] = df_address['address']

    # Write out coordinates file
    coordinates_file = os.path.join(TARGET_DATA_FOLDER, 'coordinates.geojson')
    logger.info("Writing to {}".format(coordinates_file))
    if os.path.exists(coordinates_file):
        os.remove(coordinates_file)  # fiona doesn't like to overwrite files
    df.to_file(coordinates_file, driver='GeoJSON')


def main():
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join('.', 'run.log')),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--run-all', action='store_true', default=False,
                        help='Run all of the steps: create and split image stamps, '
                             'query for addresses, and create information file for the '
                             'report. Overrides individual step flags.')
    parser.add_argument('--create-image-stamps', action='store_true', default=False,
                        help='For each building shape, creates a before and after '
                             'image stamp for the learning model, and places them '
                             'in the approriate directory (train, validation, or test)')
    parser.add_argument('--query-address-api', action='store_true', default=False,
                        help='For each building centroid, preforms a reverse '
                             'geocode query and stores the address in a cache file')
    parser.add_argument('--address-api', type=str, default='openmapquest',
                        help='Which API to use for the address query')
    parser.add_argument('--address-api-key', type=str, default=None,
                        help='Some APIs (like OpenMapQuest) require an API key')
    parser.add_argument('--create-report-info-file', action='store_true', default=False,
                        help='Creates a geojson file that contains the locations and '
                             'shapes of the buildings, their respective administrative '
                             'regions and addresses (if --query-address-api has been run)')
    args = parser.parse_args()

    # Read in the main buildings shape file
    df = geopandas.read_file(GEOJSON_FILE)
    # Remove any empty building shapes
    df = df.loc[~df['geometry'].is_empty]

    if args.create_image_stamps or args.run_all:
        createDatapoints(df)
        splitDatapoints(LABELS_FILE)

    if args.query_address_api or args.run_all:
        query_address_api(df, address_api=args.address_api, address_api_key=args.address_api_key)

    if args.create_report_info_file or args.run_all:
        create_geojson_for_visualization(df)


if __name__ == '__main__':
    main()
