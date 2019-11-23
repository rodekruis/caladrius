import React from "react";

class Table extends React.Component {
    render() {
        return (
            <div>
                <table id={this.props.table_id} className="table is-fullwidth">
                    <thead>
                        <tr>{this.renderTableHeader()}</tr>
                    </thead>
                    {this.renderTableData()}
                </table>
            </div>
        );
    }
}

export class PointInfoTable extends Table {
    renderTableData() {
        let damage = "N/A";
        let prediction = 0.0;
        let label = 0.0;
        if (this.props.selected_datum) {
            damage = this.props.selected_datum.priority;
            prediction = this.props.selected_datum.prediction
                .toString()
                .slice(0, 4);
            label = this.props.selected_datum.label.toString().slice(0, 4);
        }

        return (
            <tbody>
                <tr>
                    <td>{damage}</td>
                    <td>{prediction}</td>
                    <td>{label}</td>
                </tr>
            </tbody>
        );
    }

    renderTableHeader() {
        const header = ["Damage", "Prediction", "Label"];
        return header.map(colname => {
            return <th key={colname}>{colname}</th>;
        });
    }
}

export class CountAvgTable extends Table {
    renderTableHeader() {
        const header = ["Priority", "Low", "Medium", "High"];
        return header.map(colname => {
            return <th key={colname}>{colname}</th>;
        });
    }

    renderTableData() {
        let count_data = [0.0, 0.0, 0.0];
        let average_data = [0.0, 0.0, 0.0];
        if (this.props.data.length) {
            count_data = [
                categoryCounter(
                    this.props.data,
                    0.0,
                    this.props.damage_boundary_a
                ),
                categoryCounter(
                    this.props.data,
                    this.props.damage_boundary_a,
                    this.props.damage_boundary_b
                ),
                categoryCounter(
                    this.props.data,
                    this.props.damage_boundary_b,
                    1.0
                ),
            ];
            average_data = [
                categoryAverager(
                    this.props.data,
                    0.0,
                    this.props.damage_boundary_a
                ),
                categoryAverager(
                    this.props.data,
                    this.props.damage_boundary_a,
                    this.props.damage_boundary_b
                ),
                categoryAverager(
                    this.props.data,
                    this.props.damage_boundary_b,
                    1.0
                ),
            ];
        }
        return (
            <tbody>
                {this.renderRow("Buildings", count_data)}
                {this.renderRow("Damage", average_data)}
            </tbody>
        );
    }

    renderRow(row_name, row_data) {
        return (
            <tr>
                <th>{row_name}</th>
                <td>{row_data[0]}</td>
                <td>{row_data[1]}</td>
                <td>{row_data[2]}</td>
            </tr>
        );
    }
}

function categoryCounter(data, a, b) {
    return data.filter(function(datum) {
        return datum.prediction >= a && datum.prediction < b;
    }).length;
}

function categoryAverager(data, a, b) {
    let filterCriteria = function(datum) {
        return datum.prediction >= a && datum.prediction < b;
    };

    let length = data.filter(filterCriteria).length;

    let categoryAverage = "0.0000";

    if (length) {
        let summation = data
            .filter(filterCriteria)
            .map(function(x) {
                return Number(x.prediction);
            })
            .reduce(reducer);

        let avg = summation / length;
        categoryAverage = avg.toString().slice(0, 5);
    }

    return categoryAverage;
}

function reducer(accumulator, currentValue) {
    return accumulator + currentValue;
}
