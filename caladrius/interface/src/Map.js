import * as React from 'react';
import { Map as LeafletMap, TileLayer, Polygon, LayerGroup, LayersControl } from 'react-leaflet';
import HeatmapLayer from 'react-leaflet-heatmap-layer'
import 'leaflet/dist/leaflet.css';
import { get_point_colour } from './colours';

const map_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const attribution="&copy; <a href=&quot;http://osm.org/copyright&quot;>OpenStreetMap</a> contributors";
const zoom = 18
let center = [18.0425, -63.0548]
 

export class Map extends React.Component {

  constructor(props) {
    super(props);
    this.get_building_shape_array = this.get_building_shape_array.bind(this);
  }

  render () {
    if (Object.keys(this.props.selected_datum).length > 0) {
      center = this.props.selected_datum['feature']['geometry']['coordinates'][0][0][0]
    }

    return (
      <LeafletMap center={center} zoom={zoom} style={{height: this.props.height}}>
        <TileLayer url={map_url} attribution={attribution} />
         <LayerGroup>
         	{this.get_building_shape_array()}
        </LayerGroup>
       <LayersControl>        
          {heatMapMaker(this.props.data, 'label')}
          {heatMapMaker(this.props.data, 'prediction')}
          {heatMapMaker(this.props.data, 'category')}
        </LayersControl>
    </LeafletMap>
    )
  }

  get_building_shape_array() {
    let building_shape_array = this.props.data.map(datum => {
      let colour = get_point_colour(datum.prediction, 
        this.props.damage_boundary_a, this.props.damage_boundary_b,
        datum.objectId, this.props.selected_datum.objectId);
      return(  
        <Polygon 
        color={colour}
        positions={datum['feature']['geometry']['coordinates'][0][0]} 
        key={datum.objectId}
        onClick={() => this.props.onClick(datum)}
       />
      );
    });
    return building_shape_array
}

}


function heatMapMaker(cacheData, mode) {
  let heatCoordinates = [];
  if (mode === "label") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][0],
            x.feature.geometry.coordinates[0][0][0][1],
            x.label
        ]
    )
  } else if (mode === "prediction") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][0],
            x.feature.geometry.coordinates[0][0][0][1],
            x.prediction
        ]
    )
  } else if (mode === "category") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][0],
            x.feature.geometry.coordinates[0][0][0][1],
            categoryToValue(x.feature.properties._damage)
        ]
    )
  }
  return (  
    <LayersControl.Overlay name={mode}>
      <HeatmapLayer
            points={heatCoordinates}
            longitudeExtractor={m => m[1]}
            latitudeExtractor={m => m[0]}
            intensityExtractor={m => parseFloat(m[2])}
        />
    </LayersControl.Overlay>
  )
}

function categoryToValue(category) {
  var value
  if (category === "none") {
    value = 0
  } else if (category === "partial") {
    value = 0.2
  } else if (category === "heavy") {
    value = 0.5
  } else if (category === "destroyed") {
    value = 0.8
  } else {
    value = 0
  }
  return value
}
