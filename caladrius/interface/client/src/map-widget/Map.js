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

const MAPBOX_ACCESS_TOKEN =
    "pk.eyJ1IjoiZ3VsZmFyYXoiLCJhIjoiY2p6NW10bmxhMGRidzNldDQ1ZmwxZ2gwbCJ9.tqPa766Wzm0xwy0p9_T3Jg";
const MAPBOX_BASE_URL = "https://api.tiles.mapbox.com/v4";
const DEFAULT_MAP_ID = "mapbox.satellite";
const ATTRIBUTION =
    'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>';
const DEFAULT_ZOOM_LEVEL = 13;
const INTERACTION_ZOOM_LEVEL = 18;
let DEFAULT_CENTER_COORDINATES = [18.035, -63.07];

export class Map extends React.Component {
    constructor(props) {
        super(props);
        this.get_building_shape_array = this.get_building_shape_array.bind(
            this
        );
        this.get_admin_regions = this.get_admin_regions.bind(this);
    }

    get_building_shape_array() {
        let building_shape_array = this.props.data.map(datum => {
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
            <LayersControl.Overlay name={"Buildings"} checked={true}>
                <LayerGroup>{building_shape_array}</LayerGroup>
            </LayersControl.Overlay>
        );
    }

    get_mapbox_layer(
        map_id,
        layer_name,
        attribution,
        base_url,
        mapbox_access_token
    ) {
        const street_map_url = `${MAPBOX_BASE_URL}/${map_id}/{z}/{x}/{y}.png?access_token=${mapbox_access_token}`;
        return (
            <LayersControl.BaseLayer
                name={layer_name}
                checked={map_id === DEFAULT_MAP_ID}
            >
                <TileLayer url={street_map_url} attribution={ATTRIBUTION} />
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
                        "mapbox.streets",
                        "Street",
                        ATTRIBUTION,
                        MAPBOX_BASE_URL,
                        MAPBOX_ACCESS_TOKEN
                    )}
                    {this.get_mapbox_layer(
                        "mapbox.satellite",
                        "Satellite",
                        ATTRIBUTION,
                        MAPBOX_BASE_URL,
                        MAPBOX_ACCESS_TOKEN
                    )}
                    {this.get_admin_regions()}
                    {this.get_building_shape_array()}
                </LayersControl>
            </LeafletMap>
        );
        return map;
    }
}
