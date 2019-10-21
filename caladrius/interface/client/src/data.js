import * as d3 from "d3";
import jQuery from "jquery";
import proj4 from "proj4";

export function load_admin_regions(callback) {
    fetch("/api/dataset?name=Sint-Maarten-2017&filename=admin_regions.geojson")
        .then(res => res.json())
        .then(region_boundaries => {
            callback(
                region_boundaries["features"].map(region =>
                    region["geometry"]["coordinates"][0].map(coords =>
                        convertCoordinate(coords)
                    )
                )
            );
        });
}

function renderPredictions(predictions, callback) {
    fetch("/api/dataset?name=Sint-Maarten-2017&filename=coordinates.geojson")
        .then(res => res.json())
        .then(geoData => {
            const ssv = d3.dsvFormat(" ");
            predictions = ssv.parse(predictions);
            predictions.pop();
            predictions.forEach(function(d) {
                d.objectId = parseInt(d.filename.replace(".png", ""));
                d.label = parseFloat(d.label);
                d.prediction = parseFloat(d.prediction);
                d.category = categorizer(d.prediction);
                // feature mapping
                d.feature = getFeature(geoData, d.objectId);
                if (d.feature) {
                    d.feature.properties._damage = getFromGeo(
                        d.objectId,
                        geoData
                    );
                }
            });
            callback(predictions);
        });
}

export function load_csv_data(model_name, prediction_filename, callback) {
    const csv_path =
        "/api/model/predictions?directory=" +
        model_name +
        "&filename=" +
        prediction_filename;
    fetch(csv_path)
        .then(res => res.text())
        .then(predictions => {
            renderPredictions(predictions, callback);
        });
}

function categorizer(prediction) {
    var lowerBound = 0.3;
    var upperBound = 0.7;

    if (prediction < lowerBound) {
        return 0;
    } else if (prediction > upperBound) {
        return 2;
    } else {
        return 1;
    }
}

function getFeature(gdata, objectId) {
    var feature = null;
    for (var featureIndex in gdata.features) {
        var currentFeature = gdata.features[featureIndex];
        if (currentFeature.properties.OBJECTID === objectId) {
            var newObject = jQuery.extend(true, {}, currentFeature);
            var oldCoordinates = newObject.geometry.coordinates[0][0];
            var newGeometry = {
                coordinates: [[[]]],
            };
            for (var coordinateIndex in oldCoordinates) {
                newGeometry.coordinates[0][0].push(
                    convertCoordinate(oldCoordinates[coordinateIndex])
                );
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
        if (geo.features[i].properties.OBJECTID === number) {
            return geo.features[i].properties._damage;
        }
    }
    return "ERROR";
}

function convertCoordinate(coordinates) {
    var sourceProjection = "+proj=utm +zone=20 +datum=WGS84 +units=m +no_defs";
    var targetProjection = "+proj=longlat +datum=WGS84 +no_defs";
    let coords = proj4(sourceProjection, targetProjection, coordinates);
    return [coords[1], coords[0]];
}
