import React from 'react';
import * as d3 from "d3";
import load_csv_data from './data.js';

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

    let xScale = d3.scaleLinear().domain([0, 1]).range([0, this.props.width])
    let yScale = d3.scaleLinear().domain([0, 1]).range([0, this.props.height])

    this.state.data.then(data => {
        let u = d3.select(this.svgEl)
          .selectAll('circle')
          .data(data)

        u.enter()
          .append('circle')
          .merge(u)
          .attr('cx', d => xScale(d.label))
          .attr('cy', d => yScale(d.prediction))
          .attr('r', d => 5)

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
