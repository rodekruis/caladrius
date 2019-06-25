import * as d3 from "d3"; 
import jQuery from "jquery";
import proj4 from "proj4";
import geoData from './data/Sint-Maarten-2017/coordinates.geojson'
import csv_path from './data/Sint-Maarten-2017/test/1552303580_epoch_001_predictions.txt'

export const images_path = './data/Sint-Maarten-2017/test/'

export async function load_csv_data(){
   const allData = await Promise.all([
        d3.json(geoData),
        d3.dsv(' ', csv_path)
    ]);
    let gdata = allData[0];
    let data = allData[1];
    data.forEach(function (d) {
        d.objectId = parseInt(d.filename.replace('.png', ''));
        d.label = parseFloat(d.label);
        d.prediction = parseFloat(d.prediction);
        d.category = categorizer(d.prediction);
        // feature mapping
        d.feature = getFeature(gdata, d.objectId);
        if (d.feature) {
            d.feature.properties._damage = getFromGeo(d.objectId, gdata);
        };
    });
    data = data.filter(function (d) {
        return d.feature != null;
    });
    data.pop(); // Last element is NaN for some reason
    return data;
}

function categorizer(prediction) {
    var lowerBound = 0.3;
    var upperBound = 0.7;

    if(prediction < lowerBound) {
        return 0;
    } else if(prediction > upperBound) {
        return 2;
    } else {
        return 1;
    }
}

function getFeature(gdata, objectId) {
    var feature = null;
    for(var featureIndex in gdata.features) {
        var currentFeature = gdata.features[featureIndex];
        if(currentFeature.properties.OBJECTID === objectId) {
            var newObject = jQuery.extend(true, {}, currentFeature);
            var oldCoordinates = newObject.geometry.coordinates[0][0];
            var newGeometry = {
                'coordinates': [[[]]]
            }
            for(var coordinateIndex in oldCoordinates) {
                newGeometry.coordinates[0][0].push(convertCoordinate(oldCoordinates[coordinateIndex]));
            }
            newObject.geometry = newGeometry;
            feature = newObject;
            break;
        }
    }
    return feature;
}

function getFromGeo(number, geo) {
    for (var i in geo.features) {
        if(geo.features[i].properties.OBJECTID === number) {
            return geo.features[i].properties._damage;
        }
    }
    return 'ERROR';
}

function convertCoordinate(coordinates) {
    var sourceProjection = '+proj=utm +zone=20 +datum=WGS84 +units=m +no_defs';
    var targetProjection = '+proj=longlat +datum=WGS84 +no_defs';
    return proj4(sourceProjection, targetProjection, coordinates);
}
