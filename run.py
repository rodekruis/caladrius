import os
import sys

import torch

import rasterio
import geopandas
import json

import rasterio.mask
import rasterio.features
import rasterio.warp

TEMP_FOLDER = os.path.join('.', 'temp')

DATA_FOLDER = os.path.join('.', 'data', 'RC Challenge 1', '1')

BEFORE_FOLDER = os.path.join(DATA_FOLDER, 'Before')

AFTER_FOLDER = os.path.join(DATA_FOLDER, 'After')

GEOJSON_FOLDER = os.path.join(DATA_FOLDER, 'Building Info')

TRAINING_GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'TrainingDataset.geojson')
TEST_1_GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'TestSet_1.geojson')
TEST_2_GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'TestSet_2.geojson')
ALL_BUILDINGS_GEOJSON_FILE = os.path.join(GEOJSON_FOLDER, 'AllBuildingOutline.geojson')

print('DATA FOLDER {}'.format(os.listdir(DATA_FOLDER)))
print('BEFORE {}'.format(os.listdir(BEFORE_FOLDER)))
print('AFTER {}'.format(os.listdir(AFTER_FOLDER)))
print('GEOJSON {}'.format(os.listdir(GEOJSON_FOLDER)))

all_buildings_df = geopandas.read_file(ALL_BUILDINGS_GEOJSON_FILE)
training_df = geopandas.read_file(TRAINING_GEOJSON_FILE)
test_1_df = geopandas.read_file(TEST_1_GEOJSON_FILE)
test_2_df = geopandas.read_file(TEST_2_GEOJSON_FILE)

print(len(all_buildings_df))
print(len(training_df))
print(len(test_1_df))
print(len(test_2_df))

geometry = None

training_json = json.loads(training_df.to_json())

for index, feature in enumerate(training_json['features']):
	geometry = feature['geometry']
	if (index+1)%2 == 0:
		break

print(geometry)

BEFORE_FILE = os.path.join(BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

with rasterio.open(BEFORE_FILE) as src:
    out_image, out_transform = rasterio.mask.mask(src, [geometry], crop=True)
    print(out_image.shape)

'''
out_meta = src.meta.copy()

# save the resulting raster  
out_meta.update({
	"driver":	"GTiff",
	"height":	out_image.shape[1],
	"width":	out_image.shape[2],
	"transform":	out_transform
})

OUTPUT_FILE = os.path.join(TEMP_FOLDER, "cropped.tif")

with rasterio.open(OUTPUT_FILE, "w", **out_meta) as dest:
    dest.write(out_image)
'''
