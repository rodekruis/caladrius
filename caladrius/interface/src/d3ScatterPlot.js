import * as d3 from "d3";

const margin = { 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 };

function d3ScatterPlot(data, svg, svg_height, svg_width) {
    let width = svg_width - margin.left - margin.right
    let height = svg_height - margin.top - margin.bottom
    let xScale = d3.scaleLinear().domain([0, 1]).range([0, width])
    let yScale = d3.scaleLinear().domain([0, 1]).range([height, 0])
    let inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])

    data.then(data => {

       // Create the area for plotting
       svg = d3.select(svg)
              .append("g")
              .attr("transform", "translate(" + margin.left + "," + margin.top + ")")

       // Add the circles
       var circles = svg.selectAll('circle')
         .data(data)
       circles.enter()
          .append('circle')
          .attr('cx', d => xScale(d.prediction))
          .attr('cy', d => yScale(d.label))
          .attr('r', d => 5)
          .style("fill", function(d){
            if (inverseXScale(d3.select(this).attr('cx')) < 0.3) {
                return 'orange'
            } else if (inverseXScale(d3.select(this).attr('cx')) > 0.7) {
                return 'steelBlue'
            } else {
                return 'purple'
            }
          })
       circles.exit().remove()

       // Add the axes
       var x_axis = d3.axisBottom()
          .scale(xScale)
       var y_axis = d3.axisLeft()
          .scale(yScale)
       svg.append('g')
        .attr('transform', 'translate(0,' + (height) + ')')
        .call(d3.axisBottom(xScale))
       svg.append('g')
      .attr('transform', 'translate(0 , 0)')
      .call(d3.axisLeft(yScale).ticks(10, 's'))

       // Add the text
      svg.append('text')
      .text('Siamese network model')
      .attr('transform',
            'translate(' + (width/2) + ' ,' +
                           (-margin.top/2) + ')')
      .style('text-anchor', 'middle')
      .attr('id', 'title_thing')

      svg.append('text')
      .attr('id', 'xAxisLabel')
      .attr('transform',
            'translate(' + (width/2) + ' ,' +
                           (height + margin.top) + ')')
      .style('text-anchor', 'middle')
      .text('Predicted');

      svg.append('text')
      .attr('id', 'yAxisLabel')
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x',0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text('Actual');

    })
}



export default d3ScatterPlot;
