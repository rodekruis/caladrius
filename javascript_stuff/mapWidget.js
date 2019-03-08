function autorun()
{
  var district_polygons = []
  // var wkt = new Wkt.Wkt();

  // http://geojson.org/
  var featuresCollection = {
                              "type": "FeatureCollection",
                              "features": []
                            }
  // https://leaflet-extras.github.io/leaflet-providers/preview/
  var mymap = L.map('mapid', {renderer: L.svg()}).setView([18.02607520212528, -63.051253259181976], 17);

  // L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  //  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  //  maxZoom: 18,
  //  }).addTo(mymap);

  // L.tileLayer('https://{s}.tile.thunderforest.com/spinal-map/{z}/{x}/{y}.png?apikey=3d4a15e316d142be82541a5ac8bbd59e', {
  //  	attribution: '&copy; <a href="http://www.thunderforest.com/">Thunderforest</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  //  	apikey: '<3d4a15e316d142be82541a5ac8bbd59e>',
  //  	maxZoom: 22
  //  }).addTo(mymap);

  L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
   attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
   maxZoom:20,
   id: 'mapbox.streets',
   accessToken: 'pk.eyJ1Ijoib3Jhbmd1aCIsImEiOiJjanNxNWthYjgxMHo0NDRyMjc5MnM1c2VwIn0.oydc_gZ6NRz7H_ny4yp0Fw'
   }).addTo(mymap);


   d3.json("testGeo.json")
     .then(function(data) {
        L.geoJson(data  ,{
          style: { color: "#999", weight: 1, fillColor: "#78c679", fillOpacity: .6 },
          onEachFeature: function(feature, layer){

             // layer.bindPopup(feature["properties"]["name"])
             // console.log(layer)
             // console.log(layer["feature"])
          }
        }).addTo(mymap);
      })


}
// if (window.addEventListener) window.addEventListener("load", autorun, false);
// else if (window.attachEvent) window.attachEvent("onload", autorun);
// else window.onload = autorun;
