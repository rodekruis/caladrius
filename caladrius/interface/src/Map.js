import React from 'react';
import * as L from 'leaflet' ;
import 'leaflet/dist/leaflet.css';

const map_url = 'https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}'
const access_token = 'pk.eyJ1Ijoib3Jhbmd1aCIsImEiOiJjanNxNWthYjgxMHo0NDRyMjc5MnM1c2VwIn0.oydc_gZ6NRz7H_ny4yp0Fw'

export class Map extends React.Component{

  componentDidMount() {
    this.map = L.map('map', {
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
  }

  render() {
    return (
        <div 
        id={'map'}
        style={{'height': this.props.height, 'width': this.props.width}}
        />
    );
  }
}