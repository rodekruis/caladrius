import * as React from "react";

export class AddressList extends React.Component {
    constructor(props) {
        super(props);
        this.createAddressRows = this.createAddressRows.bind(this);
    }

    createAddressRows() {
        return this.props.data.map(datapoint => {
            return (
                <tr
                    key={datapoint.objectId}
                    className={
                        this.props.selected_datum == datapoint
                            ? "is-selected"
                            : ""
                    }
                >
                    <td>
                        <button
                            className="button is-small"
                            onClick={() => this.props.view_datapoint(datapoint)}
                            disabled={this.props.selected_datum == datapoint}
                        >
                            VIEW
                        </button>
                    </td>
                    <td>{datapoint.priority}</td>
                    <td>
                        {datapoint.feature.properties.address || (
                            <span className="has-text-danger">
                                ADDRESS NOT AVAILABLE
                            </span>
                        )}
                    </td>
                </tr>
            );
        });
    }

    createAddressTable(data) {
        return (
            <table className="table is-hoverable is-fullwidth">
                <thead>
                    <tr>
                        <th>Inspect</th>
                        <th>Priority</th>
                        <th>Address</th>
                    </tr>
                </thead>
                <tbody>{this.createAddressRows()}</tbody>
            </table>
        );
    }

    render() {
        return (
            <div className="container is-fluid">
                <h1 className="title">Address List</h1>
                {this.props.data.length > 0 ? (
                    this.createAddressTable(this.props.data)
                ) : (
                    <div class="notification">Addresses are unavailable.</div>
                )}
            </div>
        );
    }
}
