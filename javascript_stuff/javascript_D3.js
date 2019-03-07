var margins = {"right": 50, "left":50, "bottom": 50, "top": 50};
var cache = {}
var width = (window.innerWidth - 30) - (margins.left + margins.right);
var height = (window.innerHeight - 30) -(margins.top + margins.bottom);
    width = d3.min([width, height])
    height = width

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

// var csv_path = "https://oranguh.github.io/information_visualization_2019/meteo.csv"
var csv_path = "siamese_data.csv"
var csv_path = "epoch_001_predictions.txt"
// var initialized = false
load_csv_data()

function initialized(){

// initialize all the elements for later.

  d3.select("body").append("svg")
    .attr("class", "svgFullscreen")
    .attr("width", window.innerWidth - 30)
    .attr("height", window.innerHeight - 30)
  // .append("g")
  //   .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

  d3.select("body").select(".svgFullscreen").append("svg")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .append("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

  d3.select("body").select(".svgFullscreen").append("svg")
    .attr("class", "imageContainer1")
    .attr("x", d3.select("body").select(".svgContainer").attr("width"))
    .attr("width", d3.select("body").select(".svgContainer").attr("width")/2)
    .attr("height", d3.select("body").select(".svgContainer").attr("height")/2)
    .append("g")
      .attr("transform", "translate(" + margins.left + "," + margins.top + ")")
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
           .attr("xlink:href", "example.png")

     d3.select("body").select(".imageContainer1").select("g")
      .append("rect")
      .attr("class", "previewImage")
       .attr("fill", function(d){
         return "url(#" + "previewImageID1" + ")"})
     .attr("width", d3.select("body").select(".svgContainer").attr("width") - (margins.left + margins.right))
     .attr("height", d3.select("body").select(".svgContainer").attr("height") - (margins.top + margins.bottom))

   d3.select("body").select(".svgFullscreen").append("svg")
     .attr("class", "imageContainer2")
     .attr("x", d3.select("body").select(".svgContainer").attr("width") * 1.5)
     .attr("width", d3.select("body").select(".svgContainer").attr("width")/2)
     .attr("height", d3.select("body").select(".svgContainer").attr("height")/2)
     .append("g")
       .attr("transform", "translate(" + margins.left + "," + margins.top + ")")
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
            .attr("xlink:href", "example.png")

      d3.select("body").select(".imageContainer2").select("g")
       .append("rect")
       .attr("class", "previewImage")
        .attr("fill", function(d){
          return "url(#" + "previewImageID2" + ")"})
      .attr("width", d3.select("body").select(".svgContainer").attr("width") - (margins.left + margins.right))
      .attr("height", d3.select("body").select(".svgContainer").attr("height") - (margins.top + margins.bottom))
    // just implement this >.> https://bl.ocks.org/d3indepth/fabe4d1adbf658c0b73c74d3ea36d465


  var xband = d3.scaleBand()
      .domain([0, 1])
      .range([0, width])

  var yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0])

  var svgContainer = d3.select("body").select(".svgContainer").select("g")

  svgContainer.append("text")
      .text("Siamese network model")
      .attr("x", 100)
      .attr("class", "title_thing")

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
      .attr("class", "dot")

  redraw()
}
// from https://stackoverflow.com/questions/5597060/detecting-arrow-key-presses-in-javascript
window.onkeydown = checkKey;
window.addEventListener("resize", redraw);


function redraw(){
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

  d3.select("body").select(".svgFullscreen")
    .attr("class", "svgFullscreen")
    .attr("width", window.innerWidth - 30)
    .attr("height", window.innerHeight - 30)

  var svgContainer = d3.select("body").select(".svgContainer")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .select("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

  d3.select("body").select(".imageContainer1")
    .attr("x", d3.select("body").select(".svgContainer").attr("width"))
    .attr("width", d3.select("body").select(".svgContainer").attr("width")/2)
    .attr("height", d3.select("body").select(".svgContainer").attr("height")/2)
    .select("g").select("rect")
      .attr("width", (d3.select("body").select(".svgContainer").attr("width") - (margins.left + margins.right))/2)
      .attr("height", (d3.select("body").select(".svgContainer").attr("width") - (margins.bottom + margins.top))/2)

  d3.select("body").select(".imageContainer2")
    .attr("x", d3.select("body").select(".svgContainer").attr("width") * 1.5)
    .attr("width", d3.select("body").select(".svgContainer").attr("width")/2)
    .attr("height", d3.select("body").select(".svgContainer").attr("height")/2)
    .select("g").select("rect")
      .attr("width", (d3.select("body").select(".svgContainer").attr("width") - (margins.left + margins.right))/2)
      .attr("height", (d3.select("body").select(".svgContainer").attr("width") - (margins.bottom + margins.top))/2)

  svgContainer.select(".x_axis")
      .attr("transform", "translate(0," + (height) + ")")
      .call(d3.axisBottom(xScale))

  svgContainer.select(".y_axis")
      .attr("transform", "translate(0 , 0)")
      .call(d3.axisLeft(yScale).ticks(10, "s"))

  svgContainer.selectAll(".dot")
      .attr("class", "dot")
      .attr("r", 7)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .attr("fill", function(d){
        if (inverseXScale(d3.select(this).attr("cx")) < xlines[0].x1){
          return "orange"
        } else if (inverseXScale(d3.select(this).attr("cx")) > xlines[1].x1) {
          return "steelBlue"
        }
        else {
          return "purple"
        }
      })
      .on("mouseover", function(d) {
					// var xPosition = Number(d3.select(this).attr("cx")) + xScale.bandwidth / 2;
					// var yPosition = Number(d3.select(this).attr("cy")) / 2 + height / 2;
          var xPosition = width
          var yPosition = 100
          var string = "<img src= " + "example.png" + "/>"

					d3.select("#tooltip")
						.style("left", xPosition + "px")
						.style("top", yPosition + "px")
						.select("#value")
						.text(d.filename + ' ' + d.prediction + ' ' + d.label)
					d3.select("#tooltip").classed("hidden", false);
			   })
			   .on("mouseout", function() {
					d3.select("#tooltip").classed("hidden", true);
        })
      .on("click", function(d) {
        d3.select("body").select(".imageContainer1").select("g").select("#previewImageID1").select("image")
          .attr("xlink:href", "/home/meow/skool/caladrius/javascript_stuff/test/after/" + d.filename)

        d3.select("body").select(".imageContainer2").select("g").select("#previewImageID2").select("image")
          .attr("xlink:href", "/home/meow/skool/caladrius/javascript_stuff/test/before/" + d.filename)
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
      console.log(d3.select(this).attr("class").match("line_0"))
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
        console.log(d3.select(this).attr("class").match("line_0"))
        if (d3.select(this).attr("class").match("line_0")){
          return "orange"
        } else {
          return "steelBlue"
        }
      })
      .style("stroke-width", 5)
      // .call(d3.drag()
      // .on("start", dragstarted)
      // .on("drag", dragged)
      // .on("end", dragended))
      // .on("mouseover", function(d) {
      //     d3.select(this).style("cursor", "pointer");
      //   },
      //   "mouseout", function(d) {
      //     d3.select(this).style("cursor", "default");
      //   });




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

function load_csv_data(){

  // d3.csv(csv_path)
  d3.dsv(" ", csv_path).then(function(data) {
    data.forEach(function(d) {
    // console.log(d)
    d.label = +d.label;
    d.prediction = +d.prediction;
     // console.log(d);
  });
    // console.log(data)
    cache.data = data
    initialized()
  });
}
