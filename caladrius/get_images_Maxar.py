from selenium.webdriver import Firefox, Chrome
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
import urllib.request
import sys
import time
import click
import os
import glob
import fiona
import rasterio
from rasterio.windows import get_data_window
from shapely.geometry import Polygon
from tqdm import tqdm
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
from rasterio.mask import mask


def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = max(time.time() - start_time, 1)
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count * block_size * 100 / total_size), 100)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                    (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()

@click.command()
@click.option('--disaster', default='typhoon-mangkhut', help='name of the disaster')
@click.option('--dest', default='digital-globe', help='destination folder')
@click.option('--filter', default=False, help='filter images by night-time lights (yes/no)')
@click.option('--bbox', default='', help='filter images by bounding box (CSV format)')
@click.option('--maxpre', default=1000000, help='max number of pre-disaster images')
@click.option('--maxpost', default=1000000, help='max number of post-disaster images')
def main(disaster, dest, filter, bbox, maxpre, maxpost):
    # initialize webdriver
    opts = Options()
    opts.headless = True
    assert opts.headless  # operating in headless mode

    # binary = r'C:\Program Files\Mozilla Firefox\firefox.exe'
    options = Options()
    options.headless = True
    # options.binary = binary
    cap = DesiredCapabilities().FIREFOX
    cap["marionette"] = True  # optional
    browser = Firefox(options=options, capabilities=cap)#, executable_path="C:\\geckodriver\\geckodriver.exe")
    print("Headless Firefox Initialized")
    disaster = disaster.lower().replace(' ', '-')
    base_url = 'view-source:https://www.digitalglobe.com/ecosystem/open-data/'+disaster
    try:
        browser.get(base_url)
    except:
        print('ERROR:', base_url, 'not found')

    os.makedirs(dest, exist_ok=True)
    os.makedirs(dest+'/pre-event', exist_ok=True)
    os.makedirs(dest+'/post-event', exist_ok=True)

    # find all images
    image_elements = browser.find_elements_by_css_selector('a')
    image_urls = [el.get_attribute('text') for el in image_elements]
    count_pre, count_post = 0, 0
    for url in image_urls:
        name = url.split('/')[-1]
        if not name.endswith('.tif'):
            continue
        cat = url.split('/')[-2]
        name = cat+'-'+name
        if 'pre-event' in url and count_pre < maxpre:
            urllib.request.urlretrieve(url, dest+'/pre-event/'+name, reporthook)
            print('image', name, 'saved')
            count_pre += 1
        elif 'post-event' in url and count_post < maxpost:
            urllib.request.urlretrieve(url, dest+'/post-event/'+name, reporthook)
            print('image', name, 'saved')
            count_post += 1

    # filter rasters
    # if filter:
    #     with fiona.open('C:/Users/JMargutti/OneDrive - Rode Kruis/Rode Kruis/night_lights/ntl_mask/ntl_mask.shp', "r") as shapefile:
    #         shapes = [feature["geometry"] for feature in shapefile]

    print('filtering rasters')

    if bbox != '':
        bbox_tuple = [float(x) for x in bbox.split(',')]
        bbox = box(bbox_tuple[0], bbox_tuple[1], bbox_tuple[2], bbox_tuple[3])
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
        coords = getFeatures(geo)
        print('filtering on bbox:')
        print(coords)

        for raster in tqdm(glob.glob(dest + '/*/*.tif')):
            raster = raster.replace('\\', '/')
            raster_or = raster
            out_name = raster.split('.')[0] + '-cropped.tif'
            with rasterio.open(raster) as src:
                print('cropping on bbox')

                try:
                    out_img, out_transform = mask(dataset=src, shapes=coords, crop=True)
                    out_meta = src.meta.copy()
                    out_meta.update({
                        'height': out_img.shape[1],
                        'width': out_img.shape[2],
                        'transform': out_transform})

                    print('saving', out_name)
                    with rasterio.open(out_name, 'w', **out_meta) as dst:
                        dst.write(out_img)
                except:
                    print('empty raster, discard')

            os.remove(raster_or)

    # for raster in tqdm(glob.glob(dest+'/*/*.tif')):
    #     raster = raster.replace('\\', '/')
    #     raster_or = raster
    #     out_name = raster.split('.')[0] + '-cropped.tif'
    #     if 'cropped' in raster:
    #         continue

        # if filter:
        #     print('cropping on ntl')
        #     out_name_ntl = raster.split('.')[0] + '-ntl.tif'
        #     with rasterio.open(raster) as src:
        #         shapes_r = [x for x in shapes if not rasterio.coords.disjoint_bounds(src.bounds, rasterio.features.bounds(x))]
        #
        #         out_image, out_transform = rasterio.mask.mask(src, shapes_r)
        #         out_meta = src.meta
        #
        #         out_meta.update({"driver": "GTiff",
        #                          "height": out_image.shape[1],
        #                          "width": out_image.shape[2],
        #                          "transform": out_transform})
        #         # save temporary ntl file
        #         with rasterio.open(out_name_ntl, "w", **out_meta) as dst:
        #             dst.write(out_image)
        #     raster = out_name_ntl

        # with rasterio.open(raster) as src:
        #     print('cropping nan')
        #     window = get_data_window(src.read(1, masked=True))
        #
        #     kwargs = src.meta.copy()
        #     kwargs.update({
        #         'height': window.height,
        #         'width': window.width,
        #         'transform': rasterio.windows.transform(window, src.transform)})
        #
        #     print('saving', out_name)
        #     with rasterio.open(out_name, 'w', **kwargs) as dst:
        #         try:
        #             dst.write(src.read(window=window))
        #         except:
        #             print('empty raster, discard')

        # # remove temporary ntl file
        # if filter:
        #     os.remove(raster)
        # remove original raster
        # os.remove(raster_or)


if __name__ == "__main__":
    main()