import * as React from "react";
import {
    Map as LeafletMap,
    TileLayer,
    Polygon,
    LayerGroup,
    LayersControl,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import "./map.css";
import { get_prediction_colour, contrast_color_array } from "../colours";

const MAP_BASE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIBUTION =
    '&copy; <a href="http://osm.org/copyright">OpenStreetMap</a> contributors';
const DEFAULT_ZOOM_LEVEL = 13;
const INTERACTION_ZOOM_LEVEL = 18;
let DEFAULT_CENTER_COORDINATES = [18.035, -63.07];

export class Map extends React.Component {
    get_building_shape_array(key, label) {
        let building_shape_array = (
            this.props.data[key].slice(-1)[0] || []
        ).map(datum => {
            const colour = get_prediction_colour(
                datum.prediction,
                this.props.damage_boundary_a,
                this.props.damage_boundary_b
            );

            let fill_opacity = 1;
            let dash_array = 0;
            if (
                this.props.selected_datum &&
                this.props.selected_datum.object_id === datum.object_id
            ) {
                fill_opacity = 0.2;
                dash_array = 4;
            }
            return (
                <Polygon
                    color={colour}
                    weight="2"
                    positions={datum.coordinates}
                    key={datum.object_id}
                    fillOpacity={fill_opacity}
                    dashArray={dash_array}
                    onClick={() => this.props.set_datum(datum)}
                />
            );
        });
        return (
            <LayersControl.Overlay name={label} checked={true}>
                <LayerGroup>{building_shape_array}</LayerGroup>
            </LayersControl.Overlay>
        );
    }

    get_mapbox_layer(layer_name, attribution, base_url) {
        const street_map_url = base_url;
        return (
            <LayersControl.BaseLayer name={layer_name} checked={true}>
                <TileLayer url={street_map_url} attribution={attribution} />
            </LayersControl.BaseLayer>
        );
    }

    get_admin_regions() {
        const admin_boundary_array = this.props.admin_regions.map(
            (datum, index) => {
                return (
                    <Polygon
                        color={
                            contrast_color_array[
                                index % contrast_color_array.length
                            ]
                        }
                        weight="1"
                        positions={datum}
                        key={index}
                    />
                );
            }
        );
        return (
            <LayersControl.Overlay name={"Admin Regions"} checked={true}>
                <LayerGroup>{admin_boundary_array}</LayerGroup>
            </LayersControl.Overlay>
        );
    }

    render() {
        const center_coordinates = this.props.selected_datum
            ? this.props.selected_datum.coordinates[0]
            : DEFAULT_CENTER_COORDINATES;
        const zoom_level = this.props.selected_datum
            ? INTERACTION_ZOOM_LEVEL
            : DEFAULT_ZOOM_LEVEL;
        const map = (
            <LeafletMap center={center_coordinates} zoom={zoom_level}>
                <LayersControl>
                    {this.get_mapbox_layer(
                        "Open Street Map",
                        ATTRIBUTION,
                        MAP_BASE_URL
                    )}
                    {this.get_admin_regions()}
                    {this.get_building_shape_array(
                        "validation",
                        "Validation Set"
                    )}
                    {this.get_building_shape_array("test", "Test Set")}
                    {this.get_building_shape_array(
                        "inference",
                        "Inference Set"
                    )}
                </LayersControl>
            </LeafletMap>
        );
        return map;
    }
}
