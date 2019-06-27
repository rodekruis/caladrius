import React from 'react';
import * as d3 from 'd3';
import { least, partial, heavy} from './colours';

export class Table extends React.Component{

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

  drawTable() {
   if (this.ref.current) {
      let table = d3.select(this.ref.current)

      let thead = table.append('thead')
        .attr('class', 'thead-light')

      let tbody = table.append('tbody')

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

    let rows_count = tbody.select('.count') 
      .selectAll('td')
      .data([categoryCounter(that.props.data, 0, that.props.damage_boundary_a),
        categoryCounter(that.props.data, that.props.damage_boundary_a,
         that.props.damage_boundary_b),
        categoryCounter(that.props.data, that.props.damage_boundary_b, 1.0)])
    rows_count.enter()
      .append('td')
      .style('text-align', 'center')
      .text(function (d) { return d; });
    rows_count.exit().remove()
    rows_count.text(function (d) { return d; });

    let rows_avg = tbody.select('.average')
      .selectAll('td')
      .data([categoryAverager(that.props.data, 0, that.props.damage_boundary_a),
        categoryAverager(that.props.data, that.props.damage_boundary_a,
         that.props.damage_boundary_b),
        categoryAverager(that.props.data, that.props.damage_boundary_b, 1.0)])
    rows_avg.enter()
        .append('td')
        .style('text-align', 'center')
        .text(function (d) { return d; });
    rows_avg.exit().remove()
    rows_avg.text(function (d) { return d; });

   }
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
