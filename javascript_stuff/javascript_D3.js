var margins = {"right": 50, "left":50, "bottom": 50, "top": 50};
var cache = {}
var width = (window.innerWidth - 30) - (margins.left + margins.right);
var height = (window.innerHeight - 30) -(margins.top + margins.bottom);

var lines = [
  {
    'x1': 100,
    'y1': 0,
    'x2': 100,
    'y2': height
  },
  {
    'x1': 300,
    'y1': 0,
    'x2': 300,
    'y2': height
  }]
// var csv_path = "https://oranguh.github.io/information_visualization_2019/meteo.csv"
var csv_path = "siamese_data.csv"
// var initialized = false
load_csv_data()

function initialized(){

// initialize all the elements for later.

  d3.select("body").append("svg")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .append("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

    // just implement this >.> https://bl.ocks.org/d3indepth/fabe4d1adbf658c0b73c74d3ea36d465


  var xband = d3.scaleBand()
      .domain([0, 1])
      .range([0, width])

  var yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0])

  d3.select("body").select("svg").select("g")
      .append("text")
      .text("Siamese network model")
      .attr("x", 100)
      .attr("class", "title_thing")

  d3.select("svg").select("g").append("g")
      .attr("class", "x_axis")
      .attr("transform", "translate(0," + (height) + ")")
      .call(d3.axisBottom(xband))

  d3.select("svg").select("g").append("g")
      .attr("class", "y_axis")
      .attr("transform", "translate(0, 0)")
      .call(d3.axisLeft(yScale).ticks(10, "s"))

  // d3.select("body").select("svg").select("g")
  //   .append("div")
  //   .attr("class", "hidden")
  //   .attr("id", "tooltip")


  redraw()
}
// from https://stackoverflow.com/questions/5597060/detecting-arrow-key-presses-in-javascript
window.onkeydown = checkKey;
window.addEventListener("resize", redraw);


function redraw(){
  width = (window.innerWidth - 30) - margins.left - margins.right
  height = (window.innerHeight - 30) - margins.top - margins.bottom;

  width= d3.min([width, height])
  // margins.left = ((window.innerWidth - 30) - margins.left - margins.right) - width
  console.log(cache.data)


  var xValue = function(d) { return d.prediction;}, // data -> value
      xScale = d3.scaleLinear().range([0, width]), // value -> display
      xMap = function(d) { return xScale(xValue(d));}
      // , // data -> display
      // xAxis = d3.svg.axis().scale(xScale).orient("bottom");

  // setup y
  var yValue = function(d) { return d.label}, // data -> value
      yScale = d3.scaleLinear().range([height, 0]), // value -> display
      yMap = function(d) { return yScale(yValue(d));}
      // , // data -> display
      // yAxis = d3.svg.axis().scale(yScale).orient("left");

  xband = d3.scaleLinear()
      .domain([0, 1])
      .range([0, width])

  yScale = d3.scaleLinear()
      .domain([0, 1])
      .range([height, 0])

  var svg = d3.select("body").select("svg")
    .attr("class", "svgContainer")
    .attr("width", Math.round(width + margins.left + margins.right))
    .attr("height", Math.round(height + margins.top + margins.bottom))
  .select("g")
    .attr("transform", "translate(" + margins.left + "," + margins.top + ")");

  svg.select(".x_axis")
      .transition()
      .attr("transform", "translate(0," + (height) + ")")
      .call(d3.axisBottom(xband))

  svg.select(".y_axis")
      .transition()
      .attr("transform", "translate(0 , 0)")
      .call(d3.axisLeft(yScale).ticks(10, "s"))

  svg.selectAll(".dot")
      .data(cache.data)
      .enter()
      .append("circle")
      .attr("class", "dot")
      .attr("r", 3.5)
      .attr("cx", xMap)
      .attr("cy", yMap)
      .attr("fill", "orange")
      .on("mouseover", function(d) {
					var xPosition = parseFloat(d3.select(this).attr("x")) + xScale.bandwidth / 2;
					var yPosition = parseFloat(d3.select(this).attr("y")) / 2 + height / 2;
					d3.select("#tooltip")
						.style("left", xPosition + "px")
						.style("top", yPosition + "px")
						.select("#value")
						.text(d);
					d3.select("#tooltip").classed("hidden", false);
			   })
			   .on("mouseout", function() {
					d3.select("#tooltip").classed("hidden", true);
			   })


  svg.selectAll(".linedrag")
    .data(lines)
    .enter()
    .append("line")
    .attr("class", "linedrag")
    .attr("class", function(d, n){
      return "line_" + String(n)
    })
    .attr("x1", function(d){
      return d.x1
    })     // x position of the first end of the line
    .attr("y1", function(d){
      return d.y1
    })      // y position of the first end of the line
    .attr("x2", function(d){
      return d.x2
    })     // x position of the second end of the line
    .attr("y2", function(d){
      return d.y2
    })
    .style("stroke", "steelBlue")
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
        redraw()
      });

  function dragstarted(d) {
    d3.select(this).raise().classed("active", true);
  }

  function dragged(d) {
    d3.select(this).attr("x1", d.x = d3.event.x).attr("x2", d.y = d3.event.x);
  }

  function dragended(d) {
    d3.select(this).classed("active", false);
  }

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

function load_csv_data(){
  d3.csv(csv_path).then(function(data) {
    data.forEach(function(d) {
    d.label = +d.label;
    d.prediction = +d.prediction;
     // console.log(d);
  });
    console.log(data)
    cache.data = data
    initialized()
  });
}

function linspace(start, stop, num_samples) {
  // https://calebmadrigal.com/simple-d3-demos/
  return d3.range(start, stop * (num_samples / (stop-start)))
    .map(function (n) { return n / (num_samples / (stop-start)); });
};
