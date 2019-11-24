import * as React from "react";
import { Scatterplot } from "../scatter-plot/Scatterplot";
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
        };
        this.drag_threshold = this.drag_threshold.bind(this);
    }

    drag_threshold(key) {
        return (value => {
            let state_update = {};
            state_update[key] = value;
            this.setState(state_update);
            this.props.set_datum_priority(
                this.state.damage_boundary_a,
                this.state.damage_boundary_b
            );
        }).bind(this);
    }

    render() {
        return (
            <section>
                {this.props.selected_model ? (
                    <div>
                        <section className="section">
                            <div className="tile is-ancestor is-vertical">
                                <div className="tile">
                                    <div className="tile is-parent is-6">
                                        <article className="tile is-child">
                                            <Scatterplot
                                                width={600}
                                                height={600}
                                                set_datum={this.props.set_datum}
                                                onDragA={this.drag_threshold(
                                                    "damage_boundary_a"
                                                )}
                                                onDragB={this.drag_threshold(
                                                    "damage_boundary_b"
                                                )}
                                                data={this.props.data}
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
                                    <div className="tile is-vertical">
                                        <h4 id="map" className="title">
                                            Inspect Data
                                        </h4>
                                        <ImageViewer
                                            selected_datum={
                                                this.props.selected_datum
                                            }
                                        />
                                        <div className="tile">
                                            <div className="tile">
                                                <Scoreboard
                                                    selected_datum={
                                                        this.props
                                                            .selected_datum
                                                    }
                                                    data={this.props.data}
                                                    damage_boundary_a={
                                                        this.state
                                                            .damage_boundary_a
                                                    }
                                                    damage_boundary_b={
                                                        this.state
                                                            .damage_boundary_b
                                                    }
                                                    get_datum_priority={
                                                        this.props
                                                            .get_datum_priority
                                                    }
                                                />
                                            </div>
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
                    </div>
                ) : (
                    <section className="hero is-large">
                        <div className="hero-body">
                            <div className="container">
                                <h2 className="title">
                                    {this.props.render_model_selector()}
                                </h2>
                            </div>
                        </div>
                    </section>
                )}
            </section>
        );
    }
}
