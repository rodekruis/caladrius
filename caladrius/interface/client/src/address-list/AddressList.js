import * as React from "react";
import "./address-list.css";

export class AddressList extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            selected_dataset_split: "inference",
        };
    }

    set_dataset_split = split => {
        this.setState({
            selected_dataset_split: split,
        });
    };

    createAddressRows(rows) {
        return rows.map(datapoint => {
            return (
                <tr
                    key={datapoint.object_id}
                    className={
                        this.props.selected_datum === datapoint
                            ? "is-selected"
                            : ""
                    }
                >
                    <td>
                        <button
                            className="button is-small"
                            onClick={() => this.props.view_datapoint(datapoint)}
                            disabled={this.props.selected_datum === datapoint}
                        >
                            VIEW
                        </button>
                    </td>
                    <td>{this.props.get_datum_priority(datapoint)}</td>
                    <td>{datapoint.address || "ADDRESS NOT AVAILABLE"}</td>
                </tr>
            );
        });
    }

    createAddressTable() {
        const address_rows = this.props.data[
            this.state.selected_dataset_split
        ].slice(-1)[0];
        return address_rows.length ? (
            <table className="table is-hoverable is-fullwidth">
                <thead>
                    <tr>
                        <th>Inspect</th>
                        <th>Priority</th>
                        <th>Address</th>
                    </tr>
                </thead>
                <tbody>{this.createAddressRows(address_rows)}</tbody>
            </table>
        ) : (
            <div className="notification address-list-notification">
                No datapoints available.
            </div>
        );
    }

    render_dataset_split_selector() {
        return (
            <div className="tabs is-fullwidth is-boxed">
                <ul>
                    <li
                        className={
                            this.state.selected_dataset_split === "validation"
                                ? "is-active"
                                : ""
                        }
                        onClick={() => this.set_dataset_split("validation")}
                        title="Click to view validation set"
                    >
                        <a href="#/">Validation Set</a>
                    </li>
                    <li
                        className={
                            this.state.selected_dataset_split === "test"
                                ? "is-active"
                                : ""
                        }
                        onClick={() => this.set_dataset_split("test")}
                        title="Click to view test set"
                    >
                        <a href="#/">Test Set</a>
                    </li>
                    <li
                        className={
                            this.state.selected_dataset_split === "inference"
                                ? "is-active"
                                : ""
                        }
                        onClick={() => this.set_dataset_split("inference")}
                        title="Click to view inference set"
                    >
                        <a href="#/">Inference Set</a>
                    </li>
                </ul>
            </div>
        );
    }

    render() {
        return (
            <section className="section">
                <h1 className="title">Address List</h1>
                {this.render_dataset_split_selector()}
                {this.createAddressTable()}
            </section>
        );
    }
}
