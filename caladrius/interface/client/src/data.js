import * as d3 from "d3";
import jQuery from "jquery";
import proj4 from "proj4";

export function fetch_admin_regions(callback) {
    fetch("/api/Sint-Maarten-2017/admin-regions")
        .then(res => res.json())
        .then(region_boundaries => {
            if ("features" in region_boundaries) {
                region_boundaries = region_boundaries["features"].map(region =>
                    region["geometry"]["coordinates"][0].map(coordinates =>
                        convert_coordinates(coordinates)
                    )
                );
            }
            callback(region_boundaries);
        });
}

export function fetch_csv_data(model_name, callback) {
    const csv_path = "/api/" + model_name + "/predictions";
    fetch(csv_path)
        .then(res => res.json())
        .then(predictions => {
            fetch("/api/Sint-Maarten-2017")
                .then(res => res.json())
                .then(geoData => {
                    predictions = parse_predictions(predictions, geoData);
                    callback(predictions);
                });
        });
}

function parse_predictions(predictions, geoData) {
    let parsed_predictions = {
        validation: [[]],
        test: [[]],
        inference: [[]],
    };
    Object.keys(predictions).forEach(split => {
        predictions[split].forEach((epoch_predictions, epoch_index) => {
            const ssv = d3.dsvFormat(" ");
            epoch_predictions = ssv.parse(epoch_predictions);
            epoch_predictions.pop();
            epoch_predictions.forEach(function(d) {
                d.object_id = parseInt(d.filename.replace(".png", ""));
                d.label = parseFloat(d.label);
                d.prediction = parseFloat(d.prediction);
                // feature properties
                const feature = get_feature(geoData, d.object_id);
                if (feature) {
                    d.coordinates = get_coordinates(feature);
                    d.address = get_address(feature);
                }
            });
            epoch_predictions = epoch_predictions.sort((a, b) => {
                return b.prediction - a.prediction;
            });
            parsed_predictions[split][epoch_index] = epoch_predictions;
        });
    });
    return parsed_predictions;
}

function get_feature(gdata, object_id) {
    var feature = null;
    for (var featureIndex in gdata.features) {
        var currentFeature = gdata.features[featureIndex];
        if (currentFeature.properties.OBJECTID === object_id) {
            var newObject = jQuery.extend(true, {}, currentFeature);
            var oldCoordinates = newObject.geometry.coordinates[0][0];
            var newGeometry = {
                coordinates: [[[]]],
            };
            for (var coordinateIndex in oldCoordinates) {
                newGeometry.coordinates[0][0].push(
                    convert_coordinates(oldCoordinates[coordinateIndex])
                );
            }
            newObject.geometry = newGeometry;
            feature = newObject;
            break;
        }
    }
    return feature;
}

function get_coordinates(feature) {
    return feature.geometry.coordinates[0][0];
}

function get_address(feature) {
    return feature.properties.address;
}

function convert_coordinates(coordinates) {
    var sourceProjection = "+proj=utm +zone=20 +datum=WGS84 +units=m +no_defs";
    var targetProjection = "+proj=longlat +datum=WGS84 +no_defs";
    let coords = proj4(sourceProjection, targetProjection, coordinates);
    return [coords[1], coords[0]];
}
