import * as React from "react";

export class ModelList extends React.Component {
    constructor(props) {
        super(props);
    }

    createModelsRows() {
        return this.props.models.map(model => {
            return (
                <tr
                    key={model.model_name}
                    onClick={() => this.props.load_model(model)}
                    className="caladrius-clickable"
                >
                    <td>{model.model_name}</td>
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
                    </tr>
                </thead>
                <tbody>{this.createModelsRows()}</tbody>
            </table>
        );
    }

    render() {
        return (
            <section className="section">
                <h1 className="title">Model List</h1>
                {this.props.models.length > 0 ? (
                    this.createModelsTable(this.props.models)
                ) : (
                    <span>No Models Found</span>
                )}
            </section>
        );
    }
}
