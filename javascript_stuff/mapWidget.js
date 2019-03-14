function mapMaker(cacheData, mymap)
{
  // var mymap = L.map('mapid').setView([18.02607520212528, -63.051253259181976], 14);
  mymap.on("moveend", function () {
    updateMap(mymap)
  // console.log(mymap.getCenter().toString());
  });

  // mymap.on("zoomlevelschange", console.log("apa"))
  // https://leaflet-extras.github.io/leaflet-providers/preview/

  // L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  //  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  //  maxZoom: 22,
  //  }).addTo(mymap);

  // L.tileLayer('https://{s}.tile.thunderforest.com/spinal-map/{z}/{x}/{y}.png?apikey=3d4a15e316d142be82541a5ac8bbd59e', {
  //  	attribution: '&copy; <a href="http://www.thunderforest.com/">Thunderforest</a>, &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
  //  	apikey: '<3d4a15e316d142be82541a5ac8bbd59e>',
  //  	maxZoom: 22
  //  }).addTo(mymap);

  L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token={accessToken}', {
   attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, <a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, Imagery © <a href="https://www.mapbox.com/">Mapbox</a>',
   maxZoom:17,
   id: 'mapbox.streets',
   accessToken: 'pk.eyJ1Ijoib3Jhbmd1aCIsImEiOiJjanNxNWthYjgxMHo0NDRyMjc5MnM1c2VwIn0.oydc_gZ6NRz7H_ny4yp0Fw'
   }).addTo(mymap);


  L.svg({interactive:true}).addTo(mymap);

    d3.select("#mapid")
      .select("svg").attr("pointer-events", "auto")
      .select("g")
      .selectAll(".myPolygons")
      .data(cacheData)
      .enter()
      .append("polygon")
      .attr("class", "myPolygons")
      .attr("id", function(d) { return "polygon" + d.feature.properties.OBJECTID})

    d3.select("#mapid")
      .select("svg").select("g")
      .selectAll(".myPolygons")
      .attr("opacity", "0.5")
      .style("fill", function(d){
            if (d.category === 0){
            return "orange"
          } else if (d.category === 2) {
            return "steelBlue"
          }
            else {
              return "purple"
          }})
      .on("click", function(d) {
        d3.selectAll(".selectedDot").attr("class", "dot")
        d3.selectAll(".selectedPolygon").attr("class", "myPolygons")
        d3.select(this).attr("class", "myPolygons selectedPolygon")

        d3.select("#dot" + d.feature.properties.OBJECTID).attr("class", "myPolygons selectedPolygon")
        d3.select("body").select(".imageContainer1").select("g").select("#previewImageID1").select("image")
          .attr("xlink:href", "./test/after/" + d.filename)

        d3.select("body").select(".imageContainer2").select("g").select("#previewImageID2").select("image")
          .attr("xlink:href", "./test/before/" + d.filename)

        d3.select("body").select(".infoTooltipBox")
          .select('tbody').select('tr')
          .selectAll('td').remove();

        d3.select("body").select(".infoTooltipBox")
          .select('tbody').select('tr')
          .selectAll('td')
          .data([d.filename, d.prediction.toString().slice(0,4), d.label.toString().slice(0,4)])
          .enter()
            .append('td')
            .style("text-align", "center")
            .style("border", "1px solid black")
            .text(function (d) { return d; });

          updateMap(mymap)
      })
      .on("mouseover", function(d) {
        d3.select(this).style("cursor", "pointer")
          // var xPosition = Number(d3.select(this).attr("cx"))
          // var yPosition = Number(d3.select(this).attr("cy"))
         //  var xPosition = width
         //  var yPosition = 100
         //  var string = "<img src= " + "example.png" + "/>"
         //  var predictionNumber = d.prediction
         //  var labelNumber = d.label
         //  d3.select("#tooltip")
         //    .style("z-index", 100)
         //    .style("left", xPosition + "px")
         //    .style("top", yPosition + "px")
         //    .select("#value")
         //    .text("Filename: " + d.filename + " "
         //        + "Prediction: " + d.prediction.toString().slice(0,9) + " "
         //        + "Label: " + d.label.toString().slice(0,9))
         //  d3.select("#tooltip").classed("hidden", false);
         // })
         // .on("mouseout", function() {
         //  d3.select("#tooltip").classed("hidden", true);
        })
      .attr("points", function(d){
        var coords = d.feature.geometry.coordinates[0][0].map(i => mymap.latLngToLayerPoint([i[1], i[0]]))
        // console.log(coords.map(i => i.x + "," + i.y).join(" "))
        return coords.map(i => i.x + "," + i.y).join(" ")})


}
// if (window.addEventListener) window.addEventListener("load", autorun, false);
// else if (window.attachEvent) window.attachEvent("onload", autorun);
// else window.onload = autorun;

function updateMap(mymap) {
  d3.select("#mapid")
    .select("svg").select("g")
    .selectAll(".myPolygons")
    .style("fill", function(d){
          if (d3.select(this).attr("class").includes("selectedPolygon")) {
            return "red"
          }
          else
           if (d.category === 0){
          return "orange"
        } else if (d.category === 2) {
          return "steelBlue"
        } else {
          return "purple"
        }})
    .attr("points", function(d){
      var coords = d.feature.geometry.coordinates[0][0].map(i => mymap.latLngToLayerPoint([i[1], i[0]]))
      return coords.map(i => i.x + "," + i.y).join(" ")
    })
}
