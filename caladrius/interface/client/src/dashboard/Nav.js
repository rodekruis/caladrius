import * as React from "react";
import { load_csv_data } from "../data.js";
import { ModelSelector } from "./ModelSelector";
import { Report } from "../report/Report";

export class Nav extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            current_model: "",
            nav_menu_class: "",
        };
        this.load_model = this.load_model.bind(this);
        this.toggle_nav_menu_class = this.toggle_nav_menu_class.bind(this);
    }

    load_model(model) {
        const model_name = model.model_directory;
        const prediction_filename = model.predictions.test[0];
        load_csv_data(model_name, prediction_filename, data => {
            this.setState({ current_model: model });
            this.props.set_data(data);
        });
    }

    toggle_nav_menu_class() {
        this.setState({
            nav_menu_class: this.state.nav_menu_class ? "" : " is-active",
        });
    }

    render() {
        return (
            <nav
                className="navbar is-fixed-top"
                role="navigation"
                aria-label="main navigation"
            >
                <div className="navbar-brand">
                    <a className="navbar-item" href="https://www.510.global/">
                        <img src="/510-logo.png" width="74" height="24.75" />
                    </a>
                    <a className="navbar-item is-primary" href="/">
                        CALADRIUS
                    </a>

                    <a
                        role="button"
                        className={
                            "navbar-burger burger" + this.state.nav_menu_class
                        }
                        aria-label="menu"
                        aria-expanded="false"
                        data-target="navbar-caladrius-controls"
                        onClick={this.toggle_nav_menu_class}
                    >
                        <span aria-hidden="true"></span>
                        <span aria-hidden="true"></span>
                        <span aria-hidden="true"></span>
                    </a>
                </div>
                <div
                    id="navbar-caladrius-controls"
                    className={"navbar-menu" + this.state.nav_menu_class}
                >
                    <div className="navbar-end">
                        <div className="navbar-item">
                            <ModelSelector
                                current_model={this.state.current_model}
                                load_model={this.load_model}
                            />
                        </div>
                        <div className="navbar-item">
                            <Report
                                globalMap={this.state.global_map}
                                data={this.state.data}
                            />
                        </div>
                    </div>
                </div>
            </nav>
        );
    }
}
