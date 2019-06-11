import * as d3 from "d3";

const margins = { 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 };

function d3scatterplot(data, svg, svg_height, svg_width) {
    let width = svg_width - margins.left - margins.right
    let height = svg_height - margins.top - margins.bottom
    let xScale = d3.scaleLinear().domain([0, 1]).range([0, width])
    let yScale = d3.scaleLinear().domain([0, 1]).range([height, 0])
    let inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])

    data.then(data => {

        let u = d3.select(svg)
          .selectAll('circle')
          .data(data)

        u.enter()
          .append("g")
            .attr("class", "plot-space")
            .attr("transform",
      	        "translate(" + margins.left + "," + margins.top + ")"
            )
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

        u.exit().remove()
    })
}



export default d3scatterplot;
