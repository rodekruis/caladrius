import React from 'react';
import d3ScatterPlot from './d3ScatterPlot.js'

class ScatterPlot extends React.Component {

  componentDidMount() {
    this.updateChart()
  }

  componentDidUpdate() {
    this.updateChart()
  }

 updateChart() {
    d3ScatterPlot(this.props.data, this.svgEl, this.props);
 }

 render() {
    return (
            <div>
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
