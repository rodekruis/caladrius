import * as React from "react";
import { load_csv_data, load_admin_regions } from "../data.js";
import { Scatterplot } from "../scatter-plot/Scatterplot";
import { MapImage } from "../datapoint-viewer/MapImage";
import { ModelSelector } from "./ModelSelector";
import { PointInfoTable, CountAvgTable } from "../scoreboard/Tables";
import { Map } from "../map-widget/Map";
import { Report } from "../report/Report";
import { AddressList } from "../address-list/AddressList";
import "./dashboard.css";

export class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            current_model: "",
            data: [],
            admin_regions: [],
            selected_datum: {},
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7,
            map_center: [18.0425, -63.0548],
            global_map: null,
            nav_menu_class: "",
        };
        this.load_model = this.load_model.bind(this);
        this.setGlobalMap = this.setGlobalMap.bind(this);
        this.toggle_nav_menu_class = this.toggle_nav_menu_class.bind(this);
    }

    componentDidMount() {
        load_admin_regions(data => {
            this.setState({ admin_regions: data });
        });
    }

    load_model(model) {
        const model_name = model.model_directory;
        const prediction_filename = model.predictions.test[0];
        load_csv_data(model_name, prediction_filename, data => {
            this.setState({ current_model: model, data: data });
        });
    }

    handleClick(datum) {
        this.setState({
            selected_datum: datum,
        });
    }

    setGlobalMap(x) {
        this.setState({ global_map: x });
    }

    handleDragA(x) {
        this.setState({ damage_boundary_a: x });
    }

    handleDragB(x) {
        this.setState({ damage_boundary_b: x });
    }

    toggle_nav_menu_class() {
        this.setState({
            nav_menu_class: this.state.nav_menu_class ? "" : " is-active",
        });
    }

    render() {
        return (
            <div className="dashboard-container">
                <nav
                    className="navbar is-fixed-top"
                    role="navigation"
                    aria-label="main navigation"
                >
                    <div className="navbar-brand">
                        <a
                            className="navbar-item"
                            href="https://www.510.global/"
                        >
                            <img
                                src="/510-logo.png"
                                width="74"
                                height="24.75"
                            />
                        </a>
                        <a className="navbar-item is-primary" href="/">
                            CALADRIUS
                        </a>

                        <a
                            role="button"
                            className={
                                "navbar-burger burger" +
                                this.state.nav_menu_class
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
                <div className="graph-image-map-container">
                    <div className="graph-container">
                        <Scatterplot
                            width={700}
                            height={700}
                            onClick={datum => this.handleClick(datum)}
                            onDragA={x => this.handleDragA(x)}
                            onDragB={x => this.handleDragB(x)}
                            data={this.state.data}
                            selected_datum={this.state.selected_datum}
                            damage_boundary_a={this.state.damage_boundary_a}
                            damage_boundary_b={this.state.damage_boundary_b}
                        />
                    </div>
                    <div className="dashboard-container">
                        <div className="map-image-container">
                            <div className="map-before-image-container">
                                <MapImage
                                    image_label={"before"}
                                    selected_datum={this.state.selected_datum}
                                />
                            </div>
                            <div className="map-after-image-container">
                                <MapImage
                                    image_label={"after"}
                                    selected_datum={this.state.selected_datum}
                                />
                            </div>
                        </div>
                        <div className="widget-container">
                            <div className="tables-container">
                                <div className="table-selection-container">
                                    <PointInfoTable
                                        selected_datum={
                                            this.state.selected_datum
                                        }
                                        table_id={"infoToolTipBox"}
                                    />
                                </div>
                                <div className="table-global-container">
                                    <CountAvgTable
                                        data={this.state.data}
                                        damage_boundary_a={
                                            this.state.damage_boundary_a
                                        }
                                        damage_boundary_b={
                                            this.state.damage_boundary_b
                                        }
                                        table_id={"countAvgTable"}
                                    />
                                </div>
                            </div>
                            <div className="map-container">
                                <Map
                                    width={400}
                                    height={404}
                                    data={this.state.data}
                                    onClick={datum => this.handleClick(datum)}
                                    setGlobalMap={this.setGlobalMap}
                                    damage_boundary_a={
                                        this.state.damage_boundary_a
                                    }
                                    damage_boundary_b={
                                        this.state.damage_boundary_b
                                    }
                                    selected_datum={this.state.selected_datum}
                                    admin_regions={this.state.admin_regions}
                                />
                            </div>
                        </div>
                    </div>
                </div>
                {this.state.current_model ? (
                    <div className="address-container">
                        <AddressList
                            data={this.state.data}
                            view_datapoint={datum => this.handleClick(datum)}
                            selected_datum={this.state.selected_datum}
                        />
                    </div>
                ) : null}
            </div>
        );
    }
}
