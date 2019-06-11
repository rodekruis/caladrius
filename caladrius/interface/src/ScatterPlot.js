import React from 'react';
import load_csv_data from './data.js';
import d3ScatterPlot from './d3ScatterPlot.js'

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
    d3ScatterPlot(this.state.data, this.svgEl, this.props.height, this.props.width)
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
