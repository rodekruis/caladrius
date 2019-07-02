import React from 'react';
import * as L from 'leaflet' ;
import 'leaflet/dist/leaflet.css';
import 'leaflet.heat';

const map_url = 'https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}'
const access_token = 'pk.eyJ1Ijoib3Jhbmd1aCIsImEiOiJjanNxNWthYjgxMHo0NDRyMjc5MnM1c2VwIn0.oydc_gZ6NRz7H_ny4yp0Fw'

export class Map extends React.Component{

  componentDidMount() {
    this.mapobj = L.map('map', {
      center: [18.0425, -63.0548],
      zoom: 11,
      layers: [
        L.tileLayer(
          map_url, {
            'maxZoom': 17,
            'id': 'mapbox.streets',
            'accessToken': access_token 
          }
        ),
      ]
    })
     L.control.layers({}, {
        'temp': openWeatherMapo("temp_new"),
        'wind': openWeatherMapo("clouds_new"),
        'rains': openWeatherMapo("precipitation_new"),
        'clouds': openWeatherMapo("clouds_new"),
        'heat map labels': heatMapMaker(this.mapobj, this.props.data, "label"),
        'heat map prediction': heatMapMaker(this.mapobj, this.props.data, "prediction"),
        'heat map category': heatMapMaker(this.mapobj, this.props.data, "category"),
    }).addTo(this.mapobj)
  }

  render() {
    console.log(this.mapobj)
    return (
        <div 
          id={'map'}
         style={{'height': this.props.height, 'width': this.props.width}}
         >
       </div>
    );
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
