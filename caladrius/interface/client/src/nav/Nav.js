import * as React from "react";
import { Report } from "./Report";
import "./nav.css";

export class Nav extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            nav_menu_class: "",
        };
        this.toggle_nav_menu_class = this.toggle_nav_menu_class.bind(this);
    }

    toggle_nav_menu_class() {
        this.setState({
            nav_menu_class: this.state.nav_menu_class ? "" : " is-active",
        });
    }

    render() {
        return (
            <nav
                className="navbar is-fixed-top caladrius-navbar"
                role="navigation"
                aria-label="main navigation"
            >
                <div className="navbar-brand">
                    <a className="navbar-item" href="https://www.510.global/">
                        <img
                            src="/510-logo.png"
                            alt="www.510.global"
                            width="74"
                            height="24.75"
                        />
                    </a>
                    <a className="navbar-item is-primary" href="/">
                        CALADRIUS
                    </a>

                    <span
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
                    </span>
                </div>
                <div
                    id="navbar-caladrius-controls"
                    className={"navbar-menu" + this.state.nav_menu_class}
                >
                    <div className="navbar-end">
                        <div className="navbar-item">
                            {this.props.render_model_selector()}
                        </div>
                        <div className="navbar-item">
                            <Report
                                data={this.props.data}
                                selected_model={this.props.selected_model}
                                get_datum_priority={
                                    this.props.get_datum_priority
                                }
                            />
                        </div>
                    </div>
                </div>
            </nav>
        );
    }
}
