import * as React from "react";
import { EpochSelector } from "./EpochSelector";
import { ScatterPlot } from "../scatter-plot/ScatterPlot";
import { ImageViewer } from "../datapoint-viewer/ImageViewer";
import { Scoreboard } from "../scoreboard/Scoreboard";
import { Map } from "../map-widget/Map";
import { AddressList } from "../address-list/AddressList";

export class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7,
            selected_dataset_split: "test",
            epoch: this.props.data["test"].length,
        };
    }

    set_dataset_split = split => {
        this.setState({
            selected_dataset_split: split,
            epoch: this.props.data[split].length,
        });
    };

    set_epoch = epoch => {
        this.setState({ epoch: epoch });
    };

    drag_threshold(key) {
        return value => {
            let state_update = {};
            state_update[key] = value;
            this.setState(state_update);
            this.props.set_datum_priority(
                this.state.damage_boundary_a,
                this.state.damage_boundary_b
            );
        };
    }

    render() {
        return (
            <section>
                <section className="section">
                    <div className="tile is-ancestor is-vertical">
                        <div className="tabs is-fullwidth is-boxed">
                            <ul>
                                <li
                                    className={
                                        this.state.selected_dataset_split ===
                                        "validation"
                                            ? "is-active"
                                            : ""
                                    }
                                    onClick={() =>
                                        this.set_dataset_split("validation")
                                    }
                                    title="Click to view validation set"
                                >
                                    <a href="/#">Validation Set</a>
                                </li>
                                <li
                                    className={
                                        this.state.selected_dataset_split ===
                                        "test"
                                            ? "is-active"
                                            : ""
                                    }
                                    onClick={() =>
                                        this.set_dataset_split("test")
                                    }
                                    title="Click to view test set"
                                >
                                    <a href="/#">Test Set</a>
                                </li>
                            </ul>
                        </div>
                        <EpochSelector
                            epoch={this.state.epoch}
                            number_of_epochs={
                                this.props.data[
                                    this.state.selected_dataset_split
                                ].length
                            }
                            set_epoch={this.set_epoch}
                        />
                        <div className="tile">
                            <div className="tile is-parent is-6">
                                <article className="tile is-child">
                                    <ScatterPlot
                                        set_datum={this.props.set_datum}
                                        onDragA={this.drag_threshold(
                                            "damage_boundary_a"
                                        )}
                                        onDragB={this.drag_threshold(
                                            "damage_boundary_b"
                                        )}
                                        data={
                                            this.props.data[
                                                this.state
                                                    .selected_dataset_split
                                            ][this.state.epoch - 1]
                                        }
                                        selected_datum={
                                            this.props.selected_datum
                                        }
                                        damage_boundary_a={
                                            this.state.damage_boundary_a
                                        }
                                        damage_boundary_b={
                                            this.state.damage_boundary_b
                                        }
                                    />
                                </article>
                            </div>
                            <div className="tile is-vertical is-parent">
                                <h4 id="map" className="title">
                                    Inspect Data
                                </h4>
                                <ImageViewer
                                    selected_datum={this.props.selected_datum}
                                />
                                <div className="tile">
                                    <Scoreboard
                                        selected_datum={
                                            this.props.selected_datum
                                        }
                                        data={
                                            this.props.data[
                                                this.state
                                                    .selected_dataset_split
                                            ][this.state.epoch - 1]
                                        }
                                        damage_boundary_a={
                                            this.state.damage_boundary_a
                                        }
                                        damage_boundary_b={
                                            this.state.damage_boundary_b
                                        }
                                        get_datum_priority={
                                            this.props.get_datum_priority
                                        }
                                    />
                                </div>
                            </div>
                        </div>
                    </div>
                </section>
                <section className="section">
                    <h4 id="map" className="title">
                        Map
                    </h4>
                    <Map
                        data={this.props.data}
                        set_datum={this.props.set_datum}
                        damage_boundary_a={this.state.damage_boundary_a}
                        damage_boundary_b={this.state.damage_boundary_b}
                        selected_datum={this.props.selected_datum}
                        admin_regions={this.props.admin_regions}
                    />
                </section>
                <AddressList
                    data={this.props.data}
                    view_datapoint={this.props.set_datum}
                    selected_datum={this.props.selected_datum}
                    get_datum_priority={this.props.get_datum_priority}
                />
            </section>
        );
    }
}
