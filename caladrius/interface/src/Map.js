import * as React from 'react';
import 'leaflet/dist/leaflet.css';
import * as L from 'leaflet';
import { Map as LeafletMap, TileLayer, Polygon, LayerGroup} from 'react-leaflet';
import { get_point_colour, selected } from './colours';
import 'leaflet.heat';


const map_url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const attribution="&copy; <a href=&quot;http://osm.org/copyright&quot;>OpenStreetMap</a> contributors";
const zoom = 18

export class Map extends React.Component {

  render () {
 
    let polygonArray = this.props.data.map(datum => {
      let colour = get_point_colour(datum.prediction, 
        this.props.damage_boundary_a, this.props.damage_boundary_b,
        datum.objectId, this.props.selected_datum_id);
      return(  
        <Polygon 
        color={colour}
        positions={datum['feature']['geometry']['coordinates'][0][0]} 
        key={datum.objectId}
        onClick={() => this.props.onClick(datum)}
       />
      );
    });

    return (
      <LeafletMap center={this.props.map_center} zoom={zoom} style={{height: this.props.height}}>
        <TileLayer url={map_url} attribution={attribution} />
         <LayerGroup>
         	{polygonArray}
        </LayerGroup>
     </LeafletMap>
    )
  }
}

function openWeatherMapo(layer) {
  return L.tileLayer('https://tile.openweathermap.org/map/{layer}/{z}/{x}/{y}.png?appid={api_key}', {
   layer: layer,
   maxZoom:17,
   api_key: '2733c9a9c041a4ba7ce1963ae1a97dd4'
   })
}

function heatMapMaker(mymap, cacheData, mode) {
  let heatCoordinates = [];
  if (mode === "label") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][1],
            x.feature.geometry.coordinates[0][0][0][0],
            x.label
        ]
    )
  } else if (mode === "prediction") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][1],
            x.feature.geometry.coordinates[0][0][0][0],
            x.prediction
        ]
    )
  } else if (mode === "category") {
    heatCoordinates = cacheData.map(
        x => [
            x.feature.geometry.coordinates[0][0][0][1],
            x.feature.geometry.coordinates[0][0][0][0],
            categoryToValue(x.feature.properties._damage)
        ]
    )
  }
  // https://github.com/Leaflet/Leaflet.heat
  return L.heatLayer(heatCoordinates, {radius: 30, blur: 15 });
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
