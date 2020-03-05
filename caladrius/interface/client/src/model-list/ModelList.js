import * as React from "react";
import "./model-list.css";

export class ModelList extends React.Component {
    createModelsRows() {
        return this.props.models.map(model => {
            return (
                <tr
                    key={model.model_name}
                    onClick={() => this.props.load_model(model)}
                    className="caladrius-clickable"
                >
                    <td>{model.model_name}</td>
                    <td>{model.test_score}</td>
                    <td>{model.random_model_test_score}</td>
                    <td>{model.average_model_test_score}</td>
                    <td>{model.output_type}</td>
                    <td>{model.model_type}</td>
                    <td>{model.train_duration}</td>
                </tr>
            );
        });
    }

    createModelsTable() {
        return (
            <table className="table is-hoverable is-fullwidth is-striped">
                <thead>
                    <tr>
                        <th>Model Name</th>
                        <th>Test</th>
                        <th>Random</th>
                        <th>Average</th>
                        <th>Output Type</th>
                        <th>Model Type</th>
                        <th>Training Time</th>
                    </tr>
                </thead>
                <tbody>{this.createModelsRows()}</tbody>
            </table>
        );
    }

    render() {
        return (
            <section className="section model-list-section">
                <h1 className="title">Model List</h1>
                {this.props.models.length > 0 ? (
                    this.createModelsTable(this.props.models)
                ) : (
                    <div className="notification model-list-notification">
                        No models available.
                    </div>
                )}
            </section>
        );
    }
}
