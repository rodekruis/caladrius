import React from 'react';
import load_csv_data from './data.js';
import d3FullDataTable from './d3FullDataTable.js'

class FullDataTable extends React.Component {
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
    d3FullDataTable(this.state.data, this.tableEl, this.props)
 }

 render() {
    return (<div>
             <table
               width={this.props.width}
               height={this.props.height}
               ref={el => this.tableEl = el} >
             </table>
           </div>
           );
  }
}

export default FullDataTable;