import * as d3 from "d3";

var colorScheme = ['black', 'orange', 'purple', 'steelBlue'];

function d3FullDataTable(data, table, props) {

    table = d3.select(table)
      .attr('class', 'infoBox table table-bordered')

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

    data.then(data => {
      tbody.select('.count')
      .selectAll('td')
      .data([categoryCounter(data, 0),
             categoryCounter(data, 1),
             categoryCounter(data, 2)])
      .enter()
      .append('td')
      .style('text-align', 'center')
      .text(function (d) { return d; });

      tbody.select('.average')
      .selectAll('td')
      .data([categoryAverager(data, 0),
             categoryAverager(data, 1),
             categoryAverager(data, 2)])
      .enter()
      .append('td')
      .style('text-align', 'center')
      .text(function (d) { return d; });

    })

}

function categoryCounter(data, index) {
    return data.filter(datapoint => datapoint.category === index).length;
}

function categoryAverager(data, index) {
    var filterCriteria = function(datapoint) {
        return datapoint.categor === index;
    };

    let length = data.filter(filterCriteria).length;

    var categoryAverage = '0.0000';

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


export default d3FullDataTable;
