import * as React from "react";
import "./breadcrumb.css";

export class Breadcrumb extends React.Component {
    loading_crumb() {
        return (
            <li>
                <a href="/#">Loading...</a>
            </li>
        );
    }

    root_crumb() {
        return (
            <li>
                <a
                    href="/#"
                    onClick={this.props.unselect_model}
                    title="Click to view available models"
                >
                    Model List
                </a>
            </li>
        );
    }

    model_crumb() {
        return this.props.selected_model ? (
            <li>
                <a
                    href="/#"
                    onClick={this.props.unselect_model}
                    title="Click to change model"
                >
                    {this.props.selected_model.model_name}
                </a>
            </li>
        ) : (
            <li>
                <a
                    href="/#"
                    onClick={this.props.unselect_model}
                    title="Click on a row in the model list table"
                >
                    Select a Model
                </a>
            </li>
        );
    }

    datapoint_crumb() {
        return this.props.selected_model && this.props.selected_datum ? (
            <li>
                <a
                    href="/#"
                    onClick={() =>
                        this.props.set_datum(this.props.selected_datum)
                    }
                    title="Click to unselect datapoint"
                >
                    {this.props.selected_datum.object_id}
                </a>
            </li>
        ) : null;
    }

    build_crumbs() {
        return (
            <ul>
                {this.root_crumb()}
                {this.model_crumb()}
                {this.datapoint_crumb()}
            </ul>
        );
    }

    render() {
        return (
            <section className="section breadcrumb-section">
                <nav
                    className="breadcrumb has-succeeds-separator"
                    aria-label="breadcrumbs"
                >
                    {this.props.loading
                        ? this.loading_crumb()
                        : this.build_crumbs()}
                </nav>
            </section>
        );
    }
}
