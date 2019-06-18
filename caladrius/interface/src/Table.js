import React from 'react';
import d3FullDataTable from './d3FullDataTable.js'

class FullDataTable extends React.Component {

  componentDidMount() {
    this.updateChart()
  }

  componentDidUpdate() {
    this.updateChart()
  }

 updateChart() {
    d3FullDataTable(this.props.data, this.tableEl, this.props)
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