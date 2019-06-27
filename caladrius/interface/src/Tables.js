import React from 'react';
import * as d3 from 'd3';
import { least, partial, heavy} from './colours';

class Table extends React.Component {

  constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.drawTable = this.drawTable.bind(this)
  }

  componentDidMount() {
      this.drawTable()
  }

  componentDidUpdate() {
      this.fillTableValues()
  }

  render() {
    return (
      <table
        width={this.props.width}
        ref={this.ref}
        />
    )
  }
}

export class PointInfoTable extends Table { 

  drawTable() {
   if (this.ref.current) {
      let table = d3.select(this.ref.current)
        .attr('class', 'infoTooltipBox table table-bordered');

      let thead = table.append('thead')
        .attr('class', 'thead-light');

      let tbody = table.append('tbody')
        .attr('class', 'info');

      thead.selectAll('th')
        .data(['Damage', 'Prediction', 'Label'])
        .enter()
        .append('th')
          .attr('scope', 'col')
          .style('text-align', 'center')
          .text(function (d) { return d; });

      tbody.append('tr').attr('class', 'info')
      .selectAll('td')
      .data([0 , 0 , 0])
      .enter()
        .append('td')
        .style('text-align', 'center')
        .text(function (d) { return d; });
   }
 }

 fillTableValues() {

   if ((this.ref.current) && (this.props.selected_datum_id > 0)) {
    let datum = this.props.data.filter(d => 
      d.objectId === this.props.selected_datum_id)[0];
    let tbody = d3.select(this.ref.current).select('tbody');

    let row = tbody.select('tr') 
      .selectAll('td')
      .data([datum.feature.properties._damage, 
        datum.prediction.toString().slice(0,4), 
        datum.label.toString().slice(0,4)])
    row.enter()
      .append('td')
      .style('text-align', 'center')
      .text(function (d) { return d; });
    row.exit().remove();
    row.text(function (d) { return d; });
   }
  } 
}

export class CountAvgTable extends Table{

  drawTable() {
   if (this.ref.current) {
      let table = d3.select(this.ref.current);

      let thead = table.append('thead')
        .attr('class', 'thead-light');

      let tbody = table.append('tbody');

      tbody.append('tr').attr('class', 'count')
            .append('th')
            .style('text-align', 'center')
            .text('Count: ');

      tbody.append('tr').attr('class', 'average')
              .append('th')
              .style('text-align', 'center')
              .text('Average ');

      thead.selectAll('th')
        .data(['Damage:', 'Least', 'Partial', 'Heavy'])
        .enter()
        .append('th')
          .style('text-align', 'center')
          .text(function (d) { return d; })
          .style('color', function(d, i){ return colorScheme[i]});
   }
 }

 fillTableValues() {

   if (this.ref.current) {
    let that = this;
    let tbody = d3.select(this.ref.current).select('tbody');

    let row_count = tbody.select('.count') 
      .selectAll('td')
      .data([categoryCounter(that.props.data, 0, that.props.damage_boundary_a),
        categoryCounter(that.props.data, that.props.damage_boundary_a,
         that.props.damage_boundary_b),
        categoryCounter(that.props.data, that.props.damage_boundary_b, 1.0)]);
    row_count.enter()
      .append('td')
      .style('text-align', 'center')
      .text(function (d) { return d; });
    row_count.exit().remove();
    row_count.text(function (d) { return d; });

    let row_avg = tbody.select('.average')
      .selectAll('td')
      .data([categoryAverager(that.props.data, 0, that.props.damage_boundary_a),
        categoryAverager(that.props.data, that.props.damage_boundary_a,
         that.props.damage_boundary_b),
        categoryAverager(that.props.data, that.props.damage_boundary_b, 1.0)]);
    row_avg.enter()
        .append('td')
        .style('text-align', 'center')
        .text(function (d) { return d; });
    row_avg.exit().remove();
    row_avg.text(function (d) { return d; });

   }
  } 
}


function categoryCounter(data, a, b) {
  return data.filter(function(datum) {
    return ((datum.prediction >= a) && (datum.prediction < b))}).length;
 }

function categoryAverager(data, a, b) {
  let filterCriteria = function(datum) {
    return ((datum.prediction >= a) && (datum.prediction < b))
  };

  let length = data.filter(filterCriteria).length;

  let categoryAverage = '0.0000';

  if(length) {
      let summation = data.filter(filterCriteria).map(function(x) {
          return Number(x.prediction);
      }).reduce(reducer);

      let avg = summation/length;
      categoryAverage = avg.toString().slice(0,5);
  }

  return categoryAverage;
}

function reducer(accumulator, currentValue) {
  return accumulator + currentValue;
}

const colorScheme = ['black', least, partial, heavy];
