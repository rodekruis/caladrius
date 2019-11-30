import * as React from "react";

class Table extends React.Component {
    render() {
        return (
            <div>
                <table
                    id={this.props.table_id}
                    className="table is-narrow is-fullwidth"
                >
                    <thead>
                        <tr>{this.render_table_header()}</tr>
                    </thead>
                    {this.render_table_data()}
                </table>
            </div>
        );
    }
}

export class BuildingStatsTable extends Table {
    render_table_data() {
        let damage = "N/A";
        let prediction = 0.0;
        let label = 0.0;
        if (this.props.selected_datum) {
            damage = this.props.get_datum_priority(this.props.selected_datum);
            prediction = this.props.selected_datum.prediction;
            label = this.props.selected_datum.label;
        }

        prediction = parseFloat(prediction).toFixed(2);
        label = parseFloat(label).toFixed(2);

        return (
            <tbody>
                <tr>
                    <td className="has-text-centered">{damage}</td>
                    <td className="has-text-centered">{prediction}</td>
                    <td className="has-text-centered">{label}</td>
                </tr>
            </tbody>
        );
    }

    render_table_header() {
        const header = ["Class", "Prediction", "Label"];
        return header.map(colname => {
            return (
                <th key={colname} className="has-text-centered">
                    {colname}
                </th>
            );
        });
    }
}

export class ClassificationStatsTable extends Table {
    class_filter = (
        data,
        damage_boundary_a,
        damage_boundary_b,
        predcition_class
    ) => {
        return data.filter(datum => {
            const low = datum.label <= damage_boundary_a;
            const medium =
                datum.label > damage_boundary_a &&
                datum.label < damage_boundary_b;
            const high = datum.label >= damage_boundary_b;
            return this.class_condition(low, medium, high, predcition_class);
        });
    };

    class_condition(low, medium, high, predcition_class) {
        let condition_result = false;
        if (predcition_class === "low") {
            condition_result = low;
        } else if (predcition_class === "medium") {
            condition_result = medium;
        } else if (predcition_class === "high") {
            condition_result = high;
        } else if (predcition_class === "all") {
            condition_result = low || medium || high;
        }
        return condition_result;
    }

    class_count = (
        data,
        damage_boundary_a,
        damage_boundary_b,
        predcition_class
    ) => {
        return this.class_filter(
            data,
            damage_boundary_a,
            damage_boundary_b,
            predcition_class
        ).length;
    };

    class_average = (
        data,
        damage_boundary_a,
        damage_boundary_b,
        predcition_class
    ) => {
        const filtered_data = this.class_filter(
            data,
            damage_boundary_a,
            damage_boundary_b,
            predcition_class
        ).map(datum => datum.prediction);

        const total_damage = filtered_data.reduce(
            (accumulator, current_value) => {
                return accumulator + current_value;
            },
            0
        );

        let average = 0;
        if (filtered_data.length > 0) {
            average = total_damage / filtered_data.length;
        }

        average = parseFloat(average).toFixed(2);
        return average;
    };

    accuracy_condition = (
        datum,
        damage_boundary_a,
        damage_boundary_b,
        predcition_class
    ) => {
        const low =
            datum.label <= damage_boundary_a &&
            datum.prediction <= damage_boundary_a;
        const medium =
            datum.label > damage_boundary_a &&
            datum.label < damage_boundary_b &&
            datum.prediction > damage_boundary_a &&
            datum.prediction < damage_boundary_b;
        const high =
            datum.label >= damage_boundary_a &&
            datum.prediction >= damage_boundary_b;
        return this.class_condition(low, medium, high, predcition_class);
    };

    class_accuracy = (
        data,
        damage_boundary_a,
        damage_boundary_b,
        predcition_class
    ) => {
        let are_predictions_correct = [];
        this.class_filter(
            data,
            damage_boundary_a,
            damage_boundary_b,
            predcition_class
        ).map(datum => {
            are_predictions_correct.push(
                this.accuracy_condition(
                    datum,
                    damage_boundary_a,
                    damage_boundary_b,
                    predcition_class
                )
            );
        });
        let accuracy = 0;
        if (are_predictions_correct.length > 0) {
            accuracy =
                are_predictions_correct.filter(x => x).length /
                are_predictions_correct.length;
        }
        accuracy = parseFloat(accuracy).toFixed(2);
        return accuracy;
    };

    render_table_header() {
        const header = ["", "Total", "High", "Medium", "Low"];
        return header.map(colname => {
            return (
                <th key={colname} className="has-text-centered">
                    {colname}
                </th>
            );
        });
    }

    render_table_data() {
        const categories = ["all", "high", "medium", "low"];
        const calculator = {
            Accuracy: this.class_accuracy,
            Damage: this.class_average,
            Buildings: this.class_count,
        };
        return (
            <tbody>
                {Object.entries(calculator).map(([name, calculate]) => {
                    return this.render_row(
                        name,
                        categories.map(predcition_class => {
                            return calculate(
                                this.props.data,
                                this.props.damage_boundary_a,
                                this.props.damage_boundary_b,
                                predcition_class
                            );
                        })
                    );
                })}
            </tbody>
        );
    }

    render_row(row_name, row_data) {
        return (
            <tr key={row_name}>
                <th>{row_name}</th>
                {row_data.map((row, index) => (
                    <td className="has-text-centered" key={index}>
                        {row}
                    </td>
                ))}
            </tr>
        );
    }
}
