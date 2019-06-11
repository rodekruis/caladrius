import React from 'react';
import * as d3 from "d3";
import load_csv_data from './data.js';


const margins = { 'top': 50, 'right': 50, 'bottom': 50, 'left': 50 };

class ScatterPlot extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      data: load_csv_data()
    }

    this.handleClick = this.handleClick.bind(this)
    this.updateStyleAndAttrs = this.updateChart.bind(this)

  }

  handleClick() {
    this.setState({
      data: load_csv_data()
    })
  }

  componentDidMount() {
    this.updateChart()
  }

  componentDidUpdate() {
    this.updateChart()
  }

 updateChart() {

    let width = this.props.width - margins.left - margins.right
    let height = this.props.height - margins.top - margins.bottom
    let xScale = d3.scaleLinear().domain([0, 1]).range([0, width])
    let yScale = d3.scaleLinear().domain([0, 1]).range([height, 0])
    let inverseXScale = d3.scaleLinear().domain([0, width]).range([0,1])

    this.state.data.then(data => {

        let u = d3.select(this.svgEl)
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
          .call(d3.axisBottom(xScale))
          .call(d3.axisLeft(yScale))

        u.exit().remove()
    })
 }

  render() {
    return (<div>
             <svg
               width={this.props.width}
               height={this.props.height}
               ref={el => this.svgEl = el} >
             </svg>
           </div>
           );
  }
}

export default ScatterPlot;
