var margins = {"right": 50, "left":50, "bottom": 50, "top": 50};
var innerMargins = {"right": 10, "left":10, "bottom": 10, "top": 10};

var cache = {}
var width = (window.innerWidth - 30) - (margins.left + margins.right);
var height = (window.innerHeight - 30) -(margins.top + margins.bottom);
    width = d3.min([width, height])
    height = width



var mymap

var xlines = [
  {
    'x1': 0.3,
    'y1': 0,
    'x2': 0.3,
    'y2': 1
  },
  {
    'x1': 0.7,
    'y1': 0,
    'x2': 0.7,
    'y2': 1
  }]

var ylines = [
  {
    'x1': 0,
    'y1': 0.3,
    'x2': 1,
    'y2': 0.3
  },
  {
    'x1': 0,
    'y1': 0.7,
    'x2': 1,
    'y2': 0.7
  }]

var dataset_path = '../data/Sint-Maarten-2018';

var split = 'test';
var split_path = dataset_path + '/' + split;

var run_name = '1552303580';
var csv_filename = run_name + '_epoch_001_predictions.txt';
var csv_path = split_path + '/' + csv_filename;

load_csv_data(csv_path)

function initialized(){

// initialize all the elements for later.


  d3.select("body").select(".scatterPlotContainer").append("svg")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .append("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

  d3.select("body").select(".ImageOneContainer").append("svg")
    .attr("class", "imageContainer1")
    .attr("width", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width)
    .attr("height", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width)
    .append("g")
      .attr("transform", "translate(" + innerMargins.left + "," + innerMargins.top + ")")
         .append("defs")
         .append("pattern")
         .attr("id", "previewImageID1")
         .attr("width", 1)
         .attr("height", 1)
         .attr("viewBox", "0 0 100 100")
         .attr("preserveAspectRatio", "true")
          .append("image")
           .attr("width", 100)
           .attr("height", 100)
           .attr("preserveAspectRatio", "true")
           .attr("xlink:href", "images/510-logo.png")

     d3.select("body").select(".imageContainer1").select("g")
      .append("rect")
      .attr("class", "previewImage")
       .attr("fill", function(d){
         return "url(#" + "previewImageID1" + ")"})
     .attr("width", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width - (innerMargins.left + innerMargins.bottom))
     .attr("height", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width - (innerMargins.top + innerMargins.bottom))
     .attr("stroke", "black")
     .attr("stroke-width", 2)

   d3.select("body").select(".ImageTwoContainer").append("svg")
     .attr("class", "imageContainer2")
     .attr("width", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width)
     .attr("height", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width)
     .append("g")
       .attr("transform", "translate(" + innerMargins.left + "," + innerMargins.top + ")")
          .append("defs")
          .append("pattern")
          .attr("id", "previewImageID2")
          .attr("width", 1)
          .attr("height", 1)
          .attr("viewBox", "0 0 100 100")
          .attr("preserveAspectRatio", "true")
           .append("image")
            .attr("width", 100)
            .attr("height", 100)
            .attr("preserveAspectRatio", "true")
            .attr("xlink:href", "images/510-logo.png")

      d3.select("body").select(".imageContainer2").select("g")
       .append("rect")
       .attr("class", "previewImage")
        .attr("fill", function(d){
          return "url(#" + "previewImageID2" + ")"})
        .attr("width", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width - (innerMargins.left + innerMargins.bottom))
        .attr("height", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width - (innerMargins.top + innerMargins.bottom))
      .attr("stroke", "black")
      .attr("stroke-width", 2)

    d3.select("body").select(".MapContainer").append("div")
      .attr("id", "mapid")
      .style("height", "300px");
    mymap = L.map('mapid', {renderer: L.svg({interactive:true})}).setView([18.02607520212528, -63.051253259181976], 14);

    d3.select("body").select(".TableContainer").append("table")
      .attr("class", "infoBox table table-bordered")
      .append('thead')
      .attr("class", "thead-light")

    d3.select("body").select(".TableContainer").select("table")
      .append('tbody')
        .append('tr').attr("class", "count")
          .append('th')
          .style("text-align", "center")
          .text("Count: ");

    d3.select("body").select(".TableContainer").select("table")
      .select('tbody')
        .append('tr').attr("class", "average")
            .append('th')
            .style("text-align", "center")
            .text("Average ");

// tooltip infobox

    d3.select("body").select(".InfoContainer").append("table")
      .attr("class", "infoTooltipBox table table-bordered")
      .append('thead')
      .attr("class", "thead-light")
      .selectAll('th')
      .data(["Damage", "Prediction", "Label"])
      .enter()
      .append('th')
        .attr("scope", "col")
        .style("text-align", "center")
        .text(function (d) { return d; });

    d3.select("body").select(".InfoContainer").select("table")
      .append('tbody').append('tr').attr("class", "info")
      .selectAll('td')
      .data([0 , 0 , 0])
      .enter()
        .append('td')
        .style("text-align", "center")
        .text(function (d) { return d; });


    // just implement this >.> https://bl.ocks.org/d3indepth/fabe4d1adbf658c0b73c74d3ea36d465
  var xband = d3.scaleBand()
      .domain([0, 1])
      .range([0, width])

  var yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0])
  var svgContainer = d3.select("body").select(".svgContainer").select("g")

  svgContainer.append("g")
      .attr("class", "x_axis")
      .attr("transform", "translate(0," + (height) + ")")
      .call(d3.axisBottom(xband))

  svgContainer.append("g")
      .attr("class", "y_axis")
      .attr("transform", "translate(0, 0)")
      .call(d3.axisLeft(yScale).ticks(10, "s"))

  svgContainer.append("div")
    .attr("class", "hidden")
    .attr("id", "tooltip")

  svgContainer.selectAll(".xLinedrag")
    .data(xlines)
    .enter()
    .append("line")
    .attr("class", function(d, n){
      return "xLinedrag line_" + String(n)
    })
  svgContainer.selectAll(".yLinedrag")
    .data(ylines)
    .enter()
    .append("line")
    .attr("class", function(d, n){
      return "yLinedrag line_" + String(n)
    })

  svgContainer.selectAll(".dot")
      .data(cache.data)
      .enter()
      .append("circle")
      .attr("r", 7)
      .attr("class", "dot")
      .attr("id", function(d) { return "dot" + d.feature.properties.OBJECTID})
      .attr("opacity", 0.7)
      .on("mouseover", function(d) {
          d3.select(this).style("cursor", "pointer")
          d3.select(this).attr("fill", "red")
          d3.select(this).raise()
        })
      .on("mouseout", function(d) {
        d3.select(this).attr("fill", function(d){
            var inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])
                if (inverseXScale(d3.select(this).attr("cx")) < xlines[0].x1){
                  d.category = 0
                  return "orange"
                } else if (inverseXScale(d3.select(this).attr("cx")) > xlines[1].x1) {
                  d.category = 2
                  return "steelBlue"
                }
                else {
                  d.category = 1
                  return "purple"
                }
              })
      })
      .on("click", function(d) {
        console.log(d.feature.geometry.coordinates[0][0][0])
        mymap.flyTo([d.feature.geometry.coordinates[0][0][0][1], d.feature.geometry.coordinates[0][0][0][0]], 17);
        d3.selectAll(".selectedDot").attr("class", "dot")
        d3.selectAll(".selectedPolygon").attr("class", "myPolygons")
        d3.select(this).attr("class", "dot selectedDot")

        d3.select("#polygon" + d.feature.properties.OBJECTID).attr("class", "myPolygons selectedPolygon")
        // console.log("#polygon" + d.feature.properties.OBJECTID)

        d3.select("body").select(".imageContainer1").select("g").select("#previewImageID1").select("image")
          .attr("xlink:href", split_path + '/before/' + d.filename);

        d3.select("body").select(".imageContainer2").select("g").select("#previewImageID2").select("image")
          .attr("xlink:href", split_path + '/after/' + d.filename);

        d3.select("body").select(".infoTooltipBox")
          .select('tbody').select('tr')
          .selectAll('td').remove();

        d3.select("body").select(".infoTooltipBox")
          .select('tbody').select('tr')
          .selectAll('td')
          .data([d.feature.properties._damage, d.prediction.toString().slice(0,4), d.label.toString().slice(0,4)])
          .enter()
            .append('td')
            .style("text-align", "center")
            .text(function (d) { return d; });
        })

  svgContainer.append("text")
      .text("Siamese network model")
      .attr("transform",
            "translate(" + (width/2) + " ," +
                           (-margins.top/2) + ")")
      .style("text-anchor", "middle")
      .attr("id", "title_thing")

  svgContainer.append("text")
      .attr("id", "xAxisLabel")
      .attr("transform",
            "translate(" + (width/2) + " ," +
                           (height + margins.top) + ")")
      .style("text-anchor", "middle")
      .text("Predicted");


  svgContainer.append("text")
      .attr("id", "yAxisLabel")
      .attr("transform", "rotate(-90)")
      .attr("y", 0 - margins.left)
      .attr("x",0 - (height / 2))
      .attr("dy", "1em")
      .style("text-anchor", "middle")
      .text("Actual");

  redraw()
  mapMaker(cache.data, mymap)
}
// from https://stackoverflow.com/questions/5597060/detecting-arrow-key-presses-in-javascript
window.onkeydown = checkKey;
window.addEventListener("resize", redraw);

function redraw(){
  // console.log(mymap)
  width = (window.innerWidth - 30) - margins.left - margins.right
  height = (window.innerHeight - 30) - margins.top - margins.bottom;

  width = d3.min([width, height])
  height = width
  // margins.left = ((window.innerWidth - 30) - margins.left - margins.right) - width
  // console.log(cache.data)

  var xValue = function(d) { return d.prediction;}, // data -> value
      xScale = d3.scaleLinear().range([0, width]).domain([0,1]), // value -> display
      xMap = function(d) { return xScale(xValue(d));},
      inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])
  // setup y
  var yValue = function(d) { return d.label}, // data -> value
      yScale = d3.scaleLinear().range([height, 0]).domain([0,1]), // value -> display
      yMap = function(d) { return yScale(yValue(d));}

// RESCALE EVERYTHING //
console.log("hello", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().height)
console.log(d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect())
console.log(d3.select("body").select(".ImageOneContainer").node())

  var svgContainer = d3.select("body").select(".svgContainer")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .select("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

    svgContainer.select("#xAxisLabel")
        .attr("transform",
              "translate(" + (width/2) + " ," +
                             (height + margins.top) + ")")

    svgContainer.select("#yAxisLabel")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margins.left)
        .attr("x",0 - (height / 2))
        .attr("dy", "1em")

    svgContainer.select("#title_thing")
        .text("Siamese network model")
        .attr("transform",
              "translate(" + (width/2) + " ," +
                             (-margins.top/2) + ")")
        .style("text-anchor", "middle")

  d3.select("body").select(".imageContainer1")
    .attr("width", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width)
    .attr("height", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width)
    .select("g").select("rect")
      .attr("width", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width - innerMargins.left*2)
      .attr("height", d3.select("body").select(".ImageOneContainer").node().getBoundingClientRect().width - innerMargins.left*2)

  d3.select("body").select(".imageContainer2")
    .attr("width", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width)
    .attr("height", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width)
    .select("g").select("rect")
      .attr("width", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width - innerMargins.left*2)
      .attr("height", d3.select("body").select(".ImageTwoContainer").node().getBoundingClientRect().width - innerMargins.left*2)

  // d3.select("body").select("#mapWidgetContainer")

  var colorScheme = ["black", "orange", "purple", "steelBlue"]

  d3.select("body").select(".TableContainer").select(".infoBox")
      .select('thead')
      .selectAll('th')
      .data(["Damage:", "Least", "Partial", "Heavy"])
      .enter()
      .append("th")
        .style("text-align", "center")
        .text(function (d) { return d; })
        .style('color', function(d, i){ return colorScheme[i]});
    // create a row for each object in the data

d3.select("body").select(".TableContainer").select("table")
  .select('tbody').selectAll('td').remove()

d3.select("body").select(".TableContainer").select(".infoBox").select('tbody')
    .select('.count')
    .selectAll('td')
    .data([categoryCounter(cache.data, 0), categoryCounter(cache.data, 1), categoryCounter(cache.data, 2)])
    .enter()
      .append('td')
      .style("text-align", "center")
      .text(function (d) { return d; });

d3.select("body").select(".TableContainer").select(".infoBox").select('tbody')
    .select('.average')
    .selectAll('td')
    .data([categoryAverager(cache.data, 0), categoryAverager(cache.data, 1), categoryAverager(cache.data, 2)])
    .enter()
      .append('td')
      .style("text-align", "center")
      .text(function (d) { return d; });

// RESCALE OF SVG DONE //

  svgContainer.select(".x_axis")
      .attr("transform", "translate(0," + (height) + ")")
      .call(d3.axisBottom(xScale))

  svgContainer.select(".y_axis")
      .attr("transform", "translate(0 , 0)")
      .call(d3.axisLeft(yScale).ticks(10, "s"))

  svgContainer.selectAll(".dot")
      .attr("cx", xMap)
      .attr("cy", yMap)
      .attr("fill", function(d){
        if (inverseXScale(d3.select(this).attr("cx")) < xlines[0].x1){
          d.category = 0
          return "orange"
        } else if (inverseXScale(d3.select(this).attr("cx")) > xlines[1].x1) {
          d.category = 2
          return "steelBlue"
        }
        else {
          d.category = 1
          return "purple"
        }
      })



  svgContainer.selectAll(".xLinedrag")
    .attr("x1", function(d, n){
      return xScale(d.x1)
    })
    .attr("y1", function(d){
      return yScale(d.y1)
    })
    .attr("x2", function(d, n){
      return xScale(d.x2)
    })
    .attr("y2", function(d){
      return yScale(d.y2)
    })
    .style("stroke", function(d, n){
      // console.log(d3.select(this).attr("class").match("line_0"))
      if (d3.select(this).attr("class").match("line_0")){
        return "orange"
      } else {
        return "steelBlue"
      }
    })
    .style("stroke-width", 5)
    .call(d3.drag()
    .on("start", dragstarted)
    .on("drag", dragged)
    .on("end", dragended))
    .on("mouseover", function(d) {
        d3.select(this).style("cursor", "pointer");
      },
      "mouseout", function(d) {
        d3.select(this).style("cursor", "default");
      });

    svgContainer.selectAll(".yLinedrag")
      .attr("x1", function(d, n){
        return xScale(d.x1)
      })
      .attr("y1", function(d){
        return yScale(d.y1)
      })
      .attr("x2", function(d, n){
        return xScale(d.x2)
      })
      .attr("y2", function(d){
        return yScale(d.y2)
      })
      .style("stroke", function(d, n){
        // console.log(d3.select(this).attr("class").match("line_0"))
        if (d3.select(this).attr("class").match("line_0")){
          return "orange"
        } else {
          return "steelBlue"
        }
      })
      .style("stroke-width", 5)

      updateMap(mymap)
}



function checkKey(e) {
// from https://stackoverflow.com/questions/5597060/detecting-arrow-key-presses-in-javascript
    e = e || window.event;

    if (e.keyCode == '38') {
        // up arrow
      redraw()
    }
    else if (e.keyCode == '40') {
        // down arrow
      redraw()
    }
    else if (e.keyCode == '37') {
      redraw()
       // left arrow
    }
    else if (e.keyCode == '39') {
      redraw()
       // right arrow
    }
}

function dragstarted(d) {
  d3.select(this).raise().classed("active", true);
}

function dragged(d) {
  xScale = d3.scaleLinear().range([0, width]).domain([0,1])
  inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])

  n = Number(d3.select(this).attr("class").split(" ")[1].replace("line_", ""))

  if (n === 0 && (xlines[1].x1 - inverseXScale(d3.event.x)) < 0){
    xlines[n].x1 = xlines[1].x1
    xlines[n].x2 = xlines[1].x2
  } else if (n === 1 && (inverseXScale(d3.event.x) - xlines[0].x1) < 0){
    xlines[n].x1 = xlines[1].x1
    xlines[n].x2 = xlines[1].x2
  } else if (inverseXScale(d3.event.x) > 1) {
    xlines[n].x1 = 1
    xlines[n].x2 = 1
  } else if (inverseXScale(d3.event.x) < 0) {
    xlines[n].x1 = 0
    xlines[n].x2 = 0
  }  else {
    xlines[n].x1 = inverseXScale(d3.event.x)
    xlines[n].x2 = inverseXScale(d3.event.x)
  }
  redraw()
}

function dragended(d) {
  d3.select(this).classed("active", false);
}

function load_csv_data(csv_path){
    geoData = './AllBuildingOutline.geojson'
    trainingData = './TrainingDataset.geojson'

    d3.json(geoData).then(function(gdata) {
        d3.json(trainingData).then(function(tdata) {
            d3.dsv(' ', csv_path).then(function(data) {
                data.forEach(function(d, i) {
                    d.label = +d.label;
                    d.prediction = +d.prediction;
                    d.category = categorizer(d.prediction);
                    d.feature = gdata.features[Number(d.filename.replace(".png", "")) - 1];
                    if (d.feature) {
                        d.feature.properties._damage = getFromGeo(Number(d.filename.replace(".png", "")), tdata);
                    }
                });
                data.pop();
                cache.data = data;
                initialized();
            });
        });
    });
}

function categorizer(prediction){
  var lowerBound = 0.3
  var upperBound = 0.7

  if (prediction < lowerBound) {
    return  0
  } else if (prediction > upperBound) {
    return 2
  } else { return  1}

}

function categoryCounter(data, index) {
  return data.filter(datapoint => datapoint.category == index).length;
}

function categoryAverager(data, index) {
  const reducer = (accumulator, currentValue) => accumulator + currentValue;

  length = data.filter(datapoint => datapoint.category == index).length
  if (length){
    summation =  data.filter(datapoint => datapoint.category == index).map(x => Number(x.prediction)).reduce(reducer) ;
    avg = summation/length
    return avg.toString().slice(0,5)
  } else { return 0}

}

function getFromGeo(number, geo) {
  // console.log("asdasd", geo, number)

  for (i in geo.features) {
    // console.log(i, geo.features[i].properties.OBJECTID)

    // return null
    if (geo.features[i].properties.OBJECTID === number) {
      // console.log(geo.features[i].properties._damage)
      return geo.features[i].properties._damage
    }
  }
  console.log("ERROR", number, geo)
  return "ERROR"
}
