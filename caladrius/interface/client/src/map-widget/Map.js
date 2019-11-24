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

const map_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const attribution = "";
const zoom = 18;
let center = [18.0425, -63.0548];

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
            <LeafletMap center={center} zoom={zoom}>
                <TileLayer url={map_url} attribution={attribution} />
                <LayerGroup>{this.get_admin_regions()}</LayerGroup>
                <LayerGroup>{this.get_building_shape_array()}</LayerGroup>
                <LayersControl>
                    {heatMapMaker(this.props.data, "label")}
                    {heatMapMaker(this.props.data, "prediction")}
                    {heatMapMaker(this.props.data, "category")}
                </LayersControl>
            </LeafletMap>
        );
        this.props.set_global_map(map);
        return map;
    }

    render() {
        return this.state.global_map;
    }

    get_building_shape_array() {
        let building_shape_array = this.props.data.map(datum => {
            let colour = get_point_colour(
                datum.prediction,
                this.props.damage_boundary_a,
                this.props.damage_boundary_b,
                datum.objectId,
                this.props.selected_datum
                    ? this.props.selected_datum.objectId
                    : null
            );
            return (
                <Polygon
                    color={colour}
                    positions={datum.coordinates}
                    key={datum.objectId}
                    onClick={() => this.props.onClick(datum)}
                />
            );
        });
        return building_shape_array;
    }

    get_admin_regions() {
        let admin_boundary_array = this.props.admin_regions.map((datum, i) => {
            return <Polygon color={"black"} positions={datum} key={i} />;
        });
        return admin_boundary_array;
    }
}

function heatMapMaker(cacheData, mode) {
    let heatCoordinates = cacheData.map(x => [
        x.coordinates[0][0],
        x.coordinates[0][1],
        x.prediction,
    ]);
    return (
        <LayersControl.Overlay name={mode}>
            <HeatmapLayer
                points={heatCoordinates}
                longitudeExtractor={m => m[1]}
                latitudeExtractor={m => m[0]}
                intensityExtractor={m => parseFloat(m[2])}
            />
        </LayersControl.Overlay>
    );
}
