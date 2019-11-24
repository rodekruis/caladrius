import * as React from "react";
import {
    Map as LeafletMap,
    TileLayer,
    Polygon,
    LayerGroup,
    LayersControl,
} from "react-leaflet";
import HeatmapLayer from "react-leaflet-heatmap-layer";
import "leaflet/dist/leaflet.css";
import "./map.css";
import { get_point_colour } from "../colours";

const MAP_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const DEFAULT_ZOOM_LEVEL = 14;
let DEFAULT_CENTER_COORDINATES = [18.0425, -63.0548];

export class Map extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            global_map: this.initialize_map(),
        };
        this.get_building_shape_array = this.get_building_shape_array.bind(
            this
        );
        this.get_admin_regions = this.get_admin_regions.bind(this);
    }

    initialize_map() {
        if (this.props.selected_datum) {
            center = this.props.selected_datum.coordinates[0];
        }
        const map = (
            <LeafletMap
                center={DEFAULT_CENTER_COORDINATES}
                zoom={DEFAULT_ZOOM_LEVEL}
            >
                <TileLayer url={MAP_URL} />
                <LayerGroup>{this.get_building_shape_array()}</LayerGroup>
                <LayerGroup>{this.get_admin_regions()}</LayerGroup>
                <LayersControl>
                    {this.prediction_heat_map(this.props.data)}
                </LayersControl>
            </LeafletMap>
        );
        return map;
    }

    get_building_shape_array() {
        let building_shape_array = this.props.data.map(datum => {
            let colour = get_point_colour(
                datum.prediction,
                this.props.damage_boundary_a,
                this.props.damage_boundary_b,
                datum.object_id,
                this.props.selected_datum
                    ? this.props.selected_datum.object_id
                    : null
            );
            return (
                <Polygon
                    color={colour}
                    positions={datum.coordinates}
                    key={datum.object_id}
                    onClick={() => this.props.set_datum(datum)}
                />
            );
        });
        return building_shape_array;
    }

    get_admin_regions() {
        const contrast_color_array = [
            "#e6194B",
            "#3cb44b",
            "#ffe119",
            "#4363d8",
            "#f58231",
            "#42d4f4",
            "#f032e6",
            "#fabebe",
            "#469990",
            "#e6beff",
            "#9A6324",
            "#fffac8",
            "#800000",
            "#aaffc3",
            "#000075",
        ];
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
        return admin_boundary_array;
    }

    prediction_heat_map(cacheData) {
        let heatCoordinates = cacheData.map(x => [
            x.coordinates[0][0],
            x.coordinates[0][1],
            x.prediction,
        ]);
        return (
            <LayersControl.Overlay name={"Heat Map"}>
                <HeatmapLayer
                    points={heatCoordinates}
                    longitudeExtractor={m => m[1]}
                    latitudeExtractor={m => m[0]}
                    intensityExtractor={m => parseFloat(m[2])}
                />
            </LayersControl.Overlay>
        );
    }

    render() {
        return this.state.global_map;
    }
}
