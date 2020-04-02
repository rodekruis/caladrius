import overpy
import os
from gdalconst import GA_ReadOnly
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from osgeo import gdal
import re

image_folder = '../data/digital-globe/pre-event'
api = overpy.Overpass()

geopandas_dataframe = gpd.GeoDataFrame()
geopandas_dataframe['geometry'] = None

for image in os.listdir(image_folder):

    data = gdal.Open(os.path.join(image_folder, image), GA_ReadOnly)
    geoTransform = data.GetGeoTransform()
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * data.RasterXSize
    miny = maxy + geoTransform[5] * data.RasterYSize

    api = overpy.Overpass()
    query_str = '[out:json];way('+str([miny, minx, maxy, maxx])[1:-1]+')[building];' \
                '(._; >;); out;'
    query_result = api.query(query_str)

    result_ways = query_result.ways
    nodes_of_ways = [way.get_nodes(resolve_missing=True) for way in result_ways]

    list_lon_lat = []
    for way in nodes_of_ways:
        list_of_ways = []
        for node in way:
            list_of_ways.append([float(node.lon), float(node.lat)])
        list_lon_lat.append(list_of_ways)

    index_start = len(geopandas_dataframe)

    for index, way in enumerate(result_ways):
        coordinates = list_lon_lat[index]

        if coordinates[0] == coordinates[-1]:
            geopandas_dataframe.loc[index_start+index, 'geometry'] = Polygon(coordinates)
        elif len(coordinates) == 1:
            geopandas_dataframe.loc[index_start+index, 'geometry'] = Point(coordinates)
        else:
            geopandas_dataframe.loc[index_start+index, 'geometry'] = LineString(coordinates)

        id = re.compile(r'id=([0-9]+)')
        geopandas_dataframe.loc[index_start+index, 'OBJECTID'] = id.findall(str(way).strip())

geopandas_dataframe.crs = {'init': 'epsg:4326'}
name = '../data/digital-globe/AllBuildingOutline.geojson'

print('Saving as a geoJSON file as {name}'.format(name=name))
with open(name, 'w') as file:
    file.write(geopandas_dataframe.to_json())