import os, gc
import sys
import argparse
import glob
import json
import numpy as np

import pandas as pd
from pandas.io.json import json_normalize
import rasterio
from cv2 import imwrite
from tqdm import tqdm
import cv2
import rasterio.mask
import rasterio.features
import rasterio.warp
from PIL import Image
from shutil import move, rmtree, copy

import shapely.wkt
import logging

logger = logging.getLogger(__name__)
logging.getLogger("fiona").setLevel(logging.ERROR)
logging.getLogger("fiona.collection").setLevel(logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


def clahe(image_path, clip_limit=2):
    image = cv2.imread(image_path)
    # convert image to LAB color model
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # split the image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(image_lab)
    # apply CLAHE to lightness channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    # merge the CLAHE enhanced L channel with the original A and B channel
    merged_channels = cv2.merge((cl, a_channel, b_channel))
    # convert iamge from LAB color model back to RGB color model
    final_image = cv2.cvtColor(merged_channels, cv2.COLOR_LAB2BGR)
    return cv2_to_pil(final_image)


def cv2_to_pil(cv2_image):
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))


def damage_quantifier(category, label_type):
    """
    Assign value based on damage category.
    Args:
        category (str):damage category

    Returns (float): value of damage

    """
    if label_type == "classification":
        damage_dict = {
            "no-damage": 0,
            "minor-damage": 1,
            "major-damage": 2,
            "destroyed": 3,
        }
        return damage_dict[category]

    elif label_type == "regression":
        stats = {
            "none": {"mean": 0.2, "std": 0.2},
            "partial": {"mean": 0.55, "std": 0.15},
            "significant": {"mean": 0.85, "std": 0.15},
        }

        if category == "no-damage":
            value = np.random.normal(stats["none"]["mean"], stats["none"]["std"])
        elif category == "minor-damage":
            value = np.random.normal(stats["partial"]["mean"], stats["partial"]["std"])
        else:
            value = np.random.normal(
                stats["significant"]["mean"], stats["significant"]["std"]
            )

        return np.clip(value, 0.0, 1.0)


def makesquare(minx, miny, maxx, maxy, extension_factor=20):
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
        miny -= difference_range / 2
        maxy += difference_range / 2
    elif rangeX < rangeY:
        difference_range = rangeY - rangeX
        minx -= difference_range / 2
        maxx += difference_range / 2
    else:
        pass

    # update ranges
    rangeX = maxx - minx
    rangeY = maxy - miny

    # add some extra border
    minx -= rangeX / extension_factor
    maxx += rangeX / extension_factor
    miny -= rangeY / extension_factor
    maxy += rangeY / extension_factor

    return [minx, miny, maxx, maxy]

    # geoms = [
    #     {
    #         "type": "MultiPolygon",
    #         "coordinates": [
    #             [[[minx, miny], [minx, maxy], [maxx, maxy], [maxx, miny], [minx, miny]]]
    #         ],
    #     }
    # ]
    #
    # return geoms


def saveImage(image, transform, out_meta, img_path):
    """
    Saves the cropped building to a file
    Args:
        image: rasterio mask object that contains the image
        transform: transformation for mapping pixels from whole image to cropped building
        out_meta: meta information of image
        img_path (str): path where image is saved
    Returns:
        img_path (str): path where image is saved
    """
    out_meta.update(
        {
            "driver": "PNG",
            "height": image.shape[1],
            "width": image.shape[2],
            "transform": transform,
        }
    )
    with rasterio.open(img_path, "w", **out_meta) as dest:
        dest.write(image)
    return img_path


def getImage(
    source, geometry, moment, name, path_temp_data, nonzero_pixel_threshold=0.90
):
    """
    Retrieves an image and calls the function saveImage() to save the image
    Args:
        source_image (str): path where image is saved
        geometry: coordinates of bounding box that should be cropped
        moment (str): "before" or "after"
        name (str): name with which the image with the cropped building should be saved
        nonzero_pixel_threshold (float): Fraction of image pixels that must be non-zero
    """

    img_out_path = os.path.join(path_temp_data, moment, name)

    image, transform = rasterio.mask.mask(source, geometry, crop=True)
    out_meta = source.meta.copy()
    good_pixel_frac = np.count_nonzero(image) / image.size
    if np.sum(image) > 0 and good_pixel_frac > nonzero_pixel_threshold:
        return saveImage(image, transform, out_meta, img_out_path)
    return None


def splitDatapoints(
    filepath_labels,
    path_output,
    path_temp_data,
    train_split=0.8,
    validation_split=0.1,
    test_split=0.1,
):
    """
    Split the dataset in train, validation and test set and move all the images to its corresponding folder.
    Args:
        filepath_labels (str): path where labels.txt is saved, which contains the image names of all buildings and their damage score.

    Returns:

    """
    if train_split + validation_split + test_split != 1:
        logger.info("Fractions of train, validation and test set must add up to one")
        return

    with open(filepath_labels) as file:
        datapoints = file.readlines()

    allIndexes = list(range(len(datapoints)))

    # make sure training,validation and testing set are random partitions of the data
    np.random.shuffle(allIndexes)

    training_offset = int(len(allIndexes) * train_split)
    validation_offset = int(len(allIndexes) * (train_split + validation_split))

    training_indexes = allIndexes[:training_offset]
    validation_indexes = allIndexes[training_offset:validation_offset]
    testing_indexes = allIndexes[validation_offset:]

    split_mappings = {
        "train": [datapoints[i] for i in training_indexes],
        "validation": [datapoints[i] for i in validation_indexes],
        "test": [datapoints[i] for i in testing_indexes],
    }

    for split in split_mappings:
        # make directory for train, validation and test set
        split_filepath = os.path.join(path_output, split)
        os.makedirs(split_filepath, exist_ok=True)

        split_labels_file = os.path.join(split_filepath, "labels.txt")

        split_before_directory = os.path.join(split_filepath, "before")
        os.makedirs(split_before_directory, exist_ok=True)

        split_after_directory = os.path.join(split_filepath, "after")
        os.makedirs(split_after_directory, exist_ok=True)

        with open(split_labels_file, "w+") as split_file:
            for datapoint in tqdm(split_mappings[split]):
                datapoint_name = datapoint.split(" ")[0]

                before_src = os.path.join(path_temp_data, "before", datapoint_name)
                after_src = os.path.join(path_temp_data, "after", datapoint_name)

                before_dst = os.path.join(split_before_directory, datapoint_name)
                after_dst = os.path.join(split_after_directory, datapoint_name)

                # move the files from the temp folder to the final folder
                move(before_src, before_dst)
                move(after_src, after_dst)

                split_file.write(datapoint)

    # remove the folder with temporary files
    rmtree(path_temp_data)

    return split_mappings


def cropSaveImage(path_before, path_after, df_buildings, count, label_type, list_damage_types, path_temp_data,
                  labels_file, normalization):

    if normalization == "clahe":
        pilimage_pre = clahe(path_before)
        pilimage_pre.save(path_before)
        pilimage_post = clahe(path_after)
        pilimage_post.save(path_after)
    else:
        pilimage_pre = Image.open(path_before)
        pilimage_post = Image.open(path_after)

    image_pre = np.array(pilimage_pre)
    image_post = np.array(pilimage_post)

    for index, row in df_buildings.iterrows():

        # filter based on damage. Only accept described damage types. Un-classified is filtered out
        damage = row["_damage"]
        if damage not in list_damage_types:
            continue

        # pre geom
        # .bounds gives the bounding box around the polygon defined in row['geometry_pre']
        minx, miny, maxx, maxy = row["geometry_pre"].bounds
        minx, miny, maxx, maxy = makesquare(minx, miny, maxx, maxy)
        bounds_pre = [minx, miny, maxx, maxy]
        bounds_pre = [max(0, int(x)) for x in bounds_pre]

        # post geom
        minx, miny, maxx, maxy = row["geometry_post"].bounds
        minx, miny, maxx, maxy = makesquare(minx, miny, maxx, maxy)
        bounds_post = [minx, miny, maxx, maxy]
        bounds_post = [max(0, int(x)) for x in bounds_post]

        # identify data point
        objectID = row["uid"]

        crop_pre = image_pre[bounds_pre[1]:bounds_pre[3], bounds_pre[0]:bounds_pre[2]]
        before_file = os.path.join(path_temp_data, "before", "{}.png".format(objectID))
        imwrite(before_file, crop_pre)
        crop_post = image_post[bounds_post[1]:bounds_post[3], bounds_post[0]:bounds_post[2]]
        after_file = os.path.join(path_temp_data, "after", "{}.png".format(objectID))
        imwrite(after_file, crop_post)
        if (
                (before_file is not None)
                and os.path.isfile(before_file)
                and (after_file is not None)
                and os.path.isfile(after_file)
        ):
            labels_file.write(
                "{0}.png {1:.4f}\n".format(
                    objectID, damage_quantifier(damage, label_type)
                )
            )
            count += 1

    return count


def createDatapoints(df, path_images_before, path_images_after, path_temp_data,
                     label_type, list_damage_types, normalization="none"):
    """
    Loops through all the building polygons and calls functions which create an image per polygon.
    Args:
        path_temp_data:
        df (pd.DataFrame): dataframe which contains all the needed info from the labels
        path_images_before (str): path where before images are saved
        path_images_after (str): path where after images are saved
        list_damage_types (list): accepted damage types of buildings
    """

    # total number of buildings pre+post
    logger.info("Feature Size {}".format(len(df)))

    before_files = [
        os.path.join(path_images_before, before_file)
        for before_file in os.listdir(path_images_before)
    ]
    before_files.sort()
    filepath_labels = os.path.join(path_temp_data, "labels.txt")

    df_img = df[['file_pre', 'file_post']].drop_duplicates()

    with open(filepath_labels, "w+") as labels_file:
        count = 0

        for index_img, row_img in tqdm(df_img.iterrows(), total=df_img.shape[0]):

            df_buildings = df[df['file_pre'] == row_img["file_pre"]]

            count = cropSaveImage(os.path.join(path_images_before, row_img["file_pre"]),
                                  os.path.join(path_images_after, row_img["file_post"]),
                                  df_buildings, count,
                                  label_type,
                                  list_damage_types,
                                  path_temp_data,
                                  labels_file,
                                  normalization)
            gc.collect()

    logger.info("Created {} Datapoints".format(count))
    return filepath_labels


def xbd_preprocess(json_labels_path, output_folder, image_extension, disaster_names=None, disaster_types=None):
    """
    Read labels and transform to dataframe with one row per building and needed additional information
    Args:
        json_labels_path: path to folder where labels (json files) are saved
        output_folder: path to folder where to save the dataframe
        disaster_names: names of disasters to include
        disaster_types: types of disasters to include
    Returns:
        df (pd.DataFrame): dataframe containing all the polygons with related information
    """
    json_files = os.listdir(json_labels_path)

    # if we only want to take into account certain types or occurences of disasters
    # might be a faster way to do this though..
    if disaster_names:
        disaster_names_list = [item for item in disaster_names.split(",")]
        json_files_selection = [
            j for j in json_files if any(d in j for d in disaster_names_list)
        ]
        if len(json_files_selection) == 0:
            logger.info("No files match your disaster names")
    else:
        json_files_selection = json_files
    if disaster_types:
        disaster_types_list = [item for item in disaster_types.split(",")]
        json_files_selection_new = []
        for json_file in json_files_selection:
            with open(os.path.join(json_labels_path, json_file)) as d:
                data = json.load(d)
                if any(d in data['metadata']['disaster_type'] for d in disaster_types_list):
                    json_files_selection_new.append(json_file)
        json_files_selection = json_files_selection_new
        if len(json_files_selection) == 0:
            logger.info("No files match your disaster types")

    json_files_selection.sort()

    post_df = pd.DataFrame()
    pre_df = pd.DataFrame()

    for file in json_files_selection:
        json_file = os.path.join(json_labels_path, file)
        with open(json_file, "r") as f:
            data = json.load(f)

        # create one row per entry in features, xy. So one row per building
        df_temp = json_normalize(data["features"], "xy")

        # No buildings on image
        if df_temp.empty:
            continue

        # if pre file, only get coordinates for creating before image stamps
        elif "pre" in file:
            df_temp["file_pre"] = file[0:-4] + image_extension
            # wkt/geomotry_pre contains the coordinates
            df_temp = df_temp.rename(
                columns={
                    "wkt": "geometry_pre",
                    "properties.feature_type": "feature_type",
                }
            )
            pre_df = pre_df.append(
                df_temp[["geometry_pre", "file_pre"]], ignore_index=True
            )
            # continue

        # post file, get all relevant info
        elif "post" in file:
            # geometry_post is the polygon, feature_type the type of object (mostly "building"), damage_cat the
            # damage category and uid the unique id of the property
            df_temp["build_num"] = range(0, len(df_temp))
            df_temp = df_temp.rename(
                columns={
                    "wkt": "geometry_post",
                    "properties.feature_type": "feature_type",
                    "properties.subtype": "_damage",
                    "properties.uid": "uid",
                }
            )
            df_temp.insert(1, "file_post", file[0:-4] + image_extension, True)

            post_df = post_df.append(df_temp, ignore_index=True)

    # concatenate pre and post
    df = pd.concat([pre_df, post_df], axis=1)

    # wkt is a certain format to represent vector geometry and this is the format saved in the json file
    # use shapely to transform string into geometry object. With this you can e.g. calculate the area.
    if "geometry_pre" in df.columns:
        df["geometry_pre"] = df["geometry_pre"].apply(lambda x: shapely.wkt.loads(x))
    if "geometry_post" in df.columns:
        df["geometry_post"] = df["geometry_post"].apply(lambda x: shapely.wkt.loads(x))
    df.insert(
        0,
        "OBJECTID",
        df["file_post"].str.split("post").str[0] + df["build_num"].map(str),
        True,
    )

    # df.insert(0, "OBJECTID", range(0, df.shape[0]), True)

    # save the information, such that the building image names can later be related to the disaster etc.
    df.to_csv(os.path.join(output_folder, "building_information.csv"))

    return df


def create_folders(input_folder, output_folder, image_extension):

    # define before, after and label folders
    BEFORE_FOLDER = os.path.join(output_folder, "before")
    AFTER_FOLDER = os.path.join(output_folder, "after")
    JSON_FOLDER = os.path.join(input_folder, "labels")

    # make output folder
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(BEFORE_FOLDER, exist_ok=True)
    os.makedirs(AFTER_FOLDER, exist_ok=True)

    # if only a folder 'images' exists, move all images to before/after folders and delete it
    IMAGES_FOLDER = os.path.join(input_folder, "images")
    if len(os.listdir(BEFORE_FOLDER)) == 0:
        logger.info("Splitting images: images --> before")
        for file in tqdm(glob.glob(IMAGES_FOLDER+'/*_pre_*.'+image_extension)):
            if not os.path.exists(os.path.join(BEFORE_FOLDER, os.path.basename(file))):
                copy(file, BEFORE_FOLDER)
        logger.info("Splitting images: images --> after")
        for file in tqdm(glob.glob(IMAGES_FOLDER+'/*_post_*.'+image_extension)):
            if not os.path.exists(os.path.join(AFTER_FOLDER, os.path.basename(file))):
                copy(file, AFTER_FOLDER)
        #rmtree(IMAGES_FOLDER)

    # cache
    TEMP_DATA_FOLDER = os.path.join(output_folder, "temp")
    os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(TEMP_DATA_FOLDER, "before"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_DATA_FOLDER, "after"), exist_ok=True)

    return BEFORE_FOLDER, AFTER_FOLDER, JSON_FOLDER, TEMP_DATA_FOLDER


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
        default=os.path.join("../data", "xBD"),
        metavar="/path/to/dataset",
        help="Full path to the directory with /Before , /After and /labels",
    )

    parser.add_argument(
        "--image-extension",
        default="png",
        type=str,
        help="Input image extension: png or tif",
    )

    parser.add_argument(
        "--output",
        # required=True,
        default=os.path.join("../data", "xBD_buildings"),
        metavar="/path/to/output",
        help="Full path to the directory where the output should be saved",
    )

    parser.add_argument(
        "--damage",
        # required=True,
        default=["destroyed", "major-damage", "minor-damage", "no-damage"],
        metavar="damage_types",
        help="List of accepted damage types. Exclude the ones that you don't want, e.g. un-classified",
    )

    parser.add_argument(
        "--disaster-names",
        default=None,
        type=str,
        metavar="disaster_names",
        help="List of disasters to be included, as a delimited string. E.g. 'typhoon','flood'."
             "This can be types or specific occurences, as long as the json and image files contain these names.",
    )

    parser.add_argument(
        "--normalization",
        default="none",
        type=str,
        choices=["none", "clahe"],
        help="Normalize images",
    )

    parser.add_argument(
        "--disaster-types",
        default=None,
        type=str,
        metavar="disaster-types",
        help="List of disaster_types to be included, as a delimited string. E.g. 'wind', 'flooding'.",
    )

    parser.add_argument(
        "--label-type",
        default="classification",
        type=str,
        choices=["regression", "classification"],
        metavar="label_type",
        help="How the damage label should be produced, on a continuous scale or in classes.",
    )

    parser.add_argument(
        "--train",
        default=0.8,
        type=float,
        # choices=Range(0.0,1.0),
        metavar="train_split",
        help="Fraction of data that should be labelled as training data",
    )

    parser.add_argument(
        "--val",
        default=0.1,
        type=float,
        # choices=Range(0.0, 1.0),
        metavar="val_split",
        help="Fraction of data that should be labelled as training data",
    )

    parser.add_argument(
        "--test",
        default=0.1,
        type=float,
        # choices=Range(0.0, 1.0),
        metavar="test_split",
        help="Fraction of data that should be labelled as training data",
    )

    args = parser.parse_args()

    if args.create_image_stamps or args.run_all:
        logger.info("Creating training dataset.")
        BEFORE_FOLDER, AFTER_FOLDER, JSON_FOLDER, TEMP_DATA_FOLDER = create_folders(
            args.input, args.output, args.image_extension
        )
        df = xbd_preprocess(JSON_FOLDER, args.output, args.image_extension,
                            disaster_names=args.disaster_names,
                            disaster_types=args.disaster_types)
        LABELS_FILE = createDatapoints(
            df,
            BEFORE_FOLDER,
            AFTER_FOLDER,
            TEMP_DATA_FOLDER,
            args.label_type,
            args.damage,
            args.normalization
        )
        splitDatapoints(
            LABELS_FILE,
            args.output,
            TEMP_DATA_FOLDER,
            train_split=args.train,
            validation_split=args.val,
            test_split=args.test,
        )
    else:
        logger.info("Skipping creation of training dataset.")


if __name__ == "__main__":
    main()
