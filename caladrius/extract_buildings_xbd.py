import os
import sys
import argparse

import json
import numpy as np

import pandas as pd
from pandas.io.json import json_normalize
import rasterio

from tqdm import tqdm

import rasterio.mask
import rasterio.features
import rasterio.warp

from shutil import move,rmtree

import shapely.wkt
import logging

logger = logging.getLogger(__name__)
logging.getLogger("fiona").setLevel(logging.ERROR)
logging.getLogger("fiona.collection").setLevel(logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


def damage_quantifier(category,label_type):
    """
    Assign value based on damage category.
    Args:
        category (str):damage category

    Returns (float): value of damage

    """
    if label_type=="classification":
        damage_dict={"no-damage":0,"minor-damage":1,"major-damage":2,"destroyed":3}
        return damage_dict[category]

    elif label_type=="regression":
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

        if category == 'no-damage':
            value = np.random.normal(stats['none']['mean'], stats['none']['std'])
        elif category == 'minor-damage':
            value = np.random.normal(stats['partial']['mean'], stats['partial']['std'])
        else:
            value = np.random.normal(stats['significant']['mean'], stats['significant']['std'])

        return np.clip(value, 0.0, 1.0)


def makesquare(minx, miny, maxx, maxy,extension_factor=20):
    """
    Create polygon that is a square around the building and adds a certain area around the building
    Args:
        minx (float): min x coordinate of bounding box
        miny (float): min y coordinate of bounding box
        maxx (float): max x coordinate of bounding box
        maxy (float): max y coordinate of bounding box
        extension_factor (float): How much space should be added around the building. 20 refers to 5% added to each side

    Returns:
        geoms (list of dicts): geometry object with polygon coordinates
    """
    rangeX = maxx - minx
    rangeY = maxy - miny

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


def saveImage(image, transform, out_meta, folder, name,path_temp_data):
    """
    Saves the cropped building to a file
    Args:
        image: rasterio mask object that contains the image
        transform: transformation for mapping pixels from whole image to cropped building
        out_meta: meta information of image
        folder (str): which folder image is located. Either "before" or "after"
        name (str): image name

    Returns:
        file_path (str): path where image is saved
    """
    out_meta.update({
            "driver": "PNG",
            "height": image.shape[1],
            "width": image.shape[2],
            "transform": transform
        })
    directory = os.path.join(path_temp_data, folder)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, name)
    with rasterio.open(file_path, 'w', **out_meta) as dest:
        dest.write(image)
    return file_path

def getImage(source_image, geometry, moment, name, path_temp_data,nonzero_pixel_threshold=0.90):
    """
    Retrieves an image and calls the function saveImage() to save the image
    Args:
        source_image (str): path where image is saved
        geometry: coordinates of bounding box that should be cropped
        moment (str): "before" or "after"
        name (str): name with which the image with the cropped building should be saved
        nonzero_pixel_threshold (float): Fraction of image pixels that must be non-zero
    """
    with rasterio.open(source_image) as source:
        image, transform = rasterio.mask.mask(source, geometry, crop=True)
        out_meta = source.meta.copy()
        good_pixel_frac = np.count_nonzero(image) / image.size
        if np.sum(image) > 0 and good_pixel_frac > nonzero_pixel_threshold:
            return saveImage(image, transform, out_meta, moment, name,path_temp_data)
        return None

def splitDatapoints(filepath_labels,path_output,path_temp_data,train_split=0.8,validation_split=0.1,test_split=0.1):
    """
    Split the dataset in train, validation and test set and move all the images to its corresponding folder.
    Args:
        filepath_labels (str): path where labels.txt is saved, which contains the image names of all buildings and their damage score.

    Returns:

    """
    if train_split+validation_split+test_split!=1:
        logger.info('Fractions of train, validation and test set must add up to one')
        return

    with open(filepath_labels) as file:
        datapoints = file.readlines()

    allIndexes = list(range(len(datapoints)))

    #make sure training,validation and testing set are random partitions of the data
    np.random.shuffle(allIndexes)


    training_offset = int(len(allIndexes) * train_split)
    validation_offset = int(len(allIndexes) * (train_split+validation_split))

    training_indexes = allIndexes[:training_offset]
    validation_indexes = allIndexes[training_offset:validation_offset]
    testing_indexes = allIndexes[validation_offset:]

    split_mappings = {
        'train': [datapoints[i] for i in training_indexes],
        'validation': [datapoints[i] for i in validation_indexes],
        'test': [datapoints[i] for i in testing_indexes]
    }

    for split in split_mappings:
        #make directory for train, validation and test set
        split_filepath = os.path.join(path_output, split)
        os.makedirs(split_filepath, exist_ok=True)

        split_labels_file = os.path.join(split_filepath, 'labels.txt')

        split_before_directory = os.path.join(split_filepath, 'before')
        os.makedirs(split_before_directory, exist_ok=True)

        split_after_directory = os.path.join(split_filepath, 'after')
        os.makedirs(split_after_directory, exist_ok=True)

        with open(split_labels_file, 'w+') as split_file:
            for datapoint in tqdm(split_mappings[split]):
                datapoint_name = datapoint.split(' ')[0]

                before_src = os.path.join(path_temp_data, 'before', datapoint_name)
                after_src = os.path.join(path_temp_data, 'after', datapoint_name)

                before_dst = os.path.join(split_before_directory, datapoint_name)
                after_dst = os.path.join(split_after_directory, datapoint_name)

                #move the files from the temp folder to the final folder
                move(before_src, before_dst)
                move(after_src, after_dst)

                split_file.write(datapoint)

    #remove the folder with temporary files
    rmtree(path_temp_data)

    return split_mappings

def createDatapoints(df,path_images_before,path_images_after, path_temp_data,label_type,list_damage_types):
    """
    Loops through all the building polygons and calls functions which create an image per polygon.
    Args:
        path_temp_data:
        df (pd.DataFrame): dataframe which contains all the needed info from the labels
        path_images_before (str): path where before images are saved
        path_images_after (str): path where after images are saved
        list_damage_types (list): accepted damage types of buildings
    """

    #total number of buildings pre+post
    logger.info('Feature Size {}'.format(len(df)))

    before_files = [os.path.join(path_images_before, before_file) for before_file in os.listdir(path_images_before)]
    before_files.sort()
    filepath_labels=os.path.join(path_temp_data, 'labels.txt')
    with open(filepath_labels, 'w+') as labels_file:
        count = 0

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            # filter based on damage. Only accept described damage types. Un-classified is filtered out
            damage = row['_damage']
            if damage not in list_damage_types:
                continue

            # pre geom
            #.bounds gives the bounding box around the polygon defined in row['geometry_pre']
            bounds_pre = row['geometry_pre'].bounds
            geoms_pre = makesquare(*bounds_pre)

            # post geom
            bounds_post = row['geometry_post'].bounds
            geoms_post = makesquare(*bounds_post)

            # identify data point
            objectID = row['OBJECTID']

            try:
                #call function to crop the image to the building, which in turn calls function to save the cropped image
                before_file = getImage(os.path.join(path_images_before, row['file_pre']), geoms_pre,'before','{}.png'.format(objectID),path_temp_data)
                after_file = getImage(os.path.join(path_images_after, row['file_post']), geoms_post,'after', '{}.png'.format(objectID),path_temp_data)
                if (before_file is not None) and os.path.isfile(before_file) and (after_file is not None) \
                        and os.path.isfile(after_file):
                    labels_file.write('{0}.png {1:.4f}\n'.format(objectID, damage_quantifier(damage,label_type)))
                    count += 1
            except ValueError as ve:
                    continue

    logger.info('Created {} Datapoints'.format(count))
    return filepath_labels

def xbd_preprocess(json_labels_path,output_folder,disaster_types=None):
    """
    Read labels and transform to dataframe with one row per building and needed additional information
    Args:
        labels_path: path to folder where labels (json files) are saved

    Returns:
        df (pd.DataFrame): dataframe containing all the polygons with related information
    """
    json_files = os.listdir(json_labels_path)

    #if we only want to take into account certain types or occurences of disasters
    #might be a faster way to do this though..
    if disaster_types:
        disaster_types_list=[item for item in disaster_types.split(',')]
        json_files_selection=[j for j in json_files if any(d in j for d in disaster_types_list)]
        if len(json_files_selection)==0:
            logger.info('No files match your disaster types')
    else:
        json_files_selection=json_files
    json_files_selection.sort()

    post_df = pd.DataFrame()
    pre_df = pd.DataFrame()

    for file in json_files_selection:
        json_file = os.path.join(json_labels_path, file)
        with open(json_file, 'r') as f:
            data = json.load(f)

        # create one row per entry in features, xy. So one row per building
        df_temp = json_normalize(data['features'], 'xy')

        # No buildings on image
        if df_temp.empty:
            continue

        # if pre file, only get coordinates for creating before image stamps
        elif 'pre' in file:
            df_temp['file_pre'] = file[0:-4] + 'png'
            #wkt/geomotry_pre contains the coordinates
            df_temp = df_temp.rename(columns={'wkt': 'geometry_pre','properties.feature_type': 'feature_type'})
            pre_df = pre_df.append(df_temp[['geometry_pre','file_pre']], ignore_index=True)
            # continue

        # post file, get all relevant info
        elif 'post' in file:
            # geometry_post is the polygon, feature_type the type of object (mostly "building"), damage_cat the
            # damage category and uid the unique id of the property
            df_temp["build_num"] = range(0, len(df_temp))
            df_temp = df_temp.rename(
                columns={'wkt': 'geometry_post', 'properties.feature_type': 'feature_type', 'properties.subtype': '_damage',
                         'properties.uid': 'uid'})
            df_temp.insert(1, "file_post", file[0:-4] + 'png', True)

            post_df = post_df.append(df_temp, ignore_index=True)

    # concatenate pre and post
    df = pd.concat([pre_df, post_df], axis = 1)

    #wkt is a certain format to represent vector geometry and this is the format saved in the json file
    #use shapely to transform string into geometry object. With this you can e.g. calculate the area.
    if "geometry_pre" in df.columns:
        df['geometry_pre'] = df['geometry_pre'].apply(lambda x: shapely.wkt.loads(x))
    if "geometry_post" in df.columns:
        df['geometry_post'] = df['geometry_post'].apply(lambda x: shapely.wkt.loads(x))
    df.insert(0, "OBJECTID", df["file_post"].str.split("post").str[0]+df["build_num"].map(str), True)

    # df.insert(0, "OBJECTID", range(0, df.shape[0]), True)

    #save the information, such that the building image names can later be related to the disaster etc.
    df.to_csv(os.path.join(output_folder,"building_information.csv"))

    return df

def create_folders(input_folder, output_folder):

    # supported damage types. These are the xBD classification.
    # xBD also contains the category "un-classified" but we want them to be ignored, so not in this list
    DAMAGE_TYPES = ['destroyed', 'major-damage', 'minor-damage', 'no-damage']

    BEFORE_FOLDER = os.path.join(input_folder, 'Before')
    AFTER_FOLDER = os.path.join(input_folder, 'After')
    JSON_FOLDER = os.path.join(input_folder, 'labels')

    # output
    os.makedirs(output_folder, exist_ok=True)

    # cache
    TEMP_DATA_FOLDER = os.path.join(output_folder, 'temp')
    os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

    return BEFORE_FOLDER,AFTER_FOLDER,JSON_FOLDER,TEMP_DATA_FOLDER

def main():
    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(".", "run.log")),
            logging.StreamHandler(sys.stdout),
        ],
        level=logging.DEBUG,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-all",
        action="store_true",
        default=False,
        help="Run all of the steps: create and split image stamps, "
             "query for addresses, and create information file for the "
             "report. Overrides individual step flags.",
    )
    parser.add_argument(
        "--create-image-stamps",
        action="store_true",
        default=False,
        help="For each building shape, creates a before and after "
             "image stamp for the learning model, and places them "
             "in the approriate directory (train, validation, or test)",
    )

    parser.add_argument(
        "--input",
        # required=True,
        default=os.path.join('../data', 'xBD'),
        metavar="/path/to/dataset",
        help="Full path to the directory with /Before , /After and /labels",
    )

    parser.add_argument(
        "--output",
        # required=True,
        default=os.path.join('../data', 'xBD_buildings'),
        metavar="/path/to/output",
        help="Full path to the directory where the output should be saved",
    )


    parser.add_argument(
        "--damage",
        # required=True,
        default=['destroyed', 'major-damage', 'minor-damage', 'no-damage'],
        metavar="damage_types",
        help="List of accepted damage types. Exclude the ones that you don't want, e.g. un-classified",
    )

    parser.add_argument(
        "--disaster",
        default=None,
        type=str,
        metavar="disaster_types",
        help="List of disasters to be included, as a delimited string. E.g. 'typhoon','flood' This can be types or specific occurences, as long as the json and image files contain these names."
    )

    parser.add_argument(
        "--label-type",
        default="regression",
        type=str,
        choices=["regression","classification"],
        metavar="label_type",
        help="How the damage label should be produced, on a continuous scale or in classes."
    )

    parser.add_argument(
        "--train",
        default=0.8,
        type=float,
        # choices=Range(0.0,1.0),
        metavar="train_split",
        help="Fraction of data that should be labelled as training data"
    )

    parser.add_argument(
        "--val",
        default=0.1,
        type=float,
        # choices=Range(0.0, 1.0),
        metavar="val_split",
        help="Fraction of data that should be labelled as training data"
    )

    parser.add_argument(
        "--test",
        default=0.1,
        type=float,
        # choices=Range(0.0, 1.0),
        metavar="test_split",
        help="Fraction of data that should be labelled as training data"
    )

    if args.create_image_stamps or args.run_all:
        logger.info("Creating training dataset.")
        BEFORE_FOLDER, AFTER_FOLDER, JSON_FOLDER, TEMP_DATA_FOLDER = create_folders(args.input, args.output)
        df = xbd_preprocess(JSON_FOLDER, args.output, disaster_types=args.disaster)
        LABELS_FILE = createDatapoints(df, BEFORE_FOLDER, AFTER_FOLDER, TEMP_DATA_FOLDER, args.label_type, args.damage)
        splitDatapoints(LABELS_FILE, args.output, TEMP_DATA_FOLDER,train_split=args.train,validation_split=args.val,test_split=args.test)
    else:
        logger.info("Skipping creation of training dataset.")

if __name__ == '__main__':
    main()