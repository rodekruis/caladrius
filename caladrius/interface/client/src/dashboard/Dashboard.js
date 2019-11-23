import * as React from "react";
import { Scatterplot } from "../scatter-plot/Scatterplot";
import { ImageViewer } from "../datapoint-viewer/ImageViewer";
import { Scoreboard } from "../scoreboard/Scoreboard";
import { Map } from "../map-widget/Map";

export class Dashboard extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7,
            global_map: null,
        };
        this.set_global_map = this.set_global_map.bind(this);
    }

    set_global_map(x) {
        this.setState({ global_map: x });
    }

    handleDragA(x) {
        this.setState({ damage_boundary_a: x });
    }

    handleDragB(x) {
        this.setState({ damage_boundary_b: x });
    }

    render() {
        return (
            <section className="section">
                {this.props.selected_model ? (
                    <div>
                        <div className="tile is-ancestor is-vertical">
                            <div className="tile">
                                <div className="tile is-parent is-6">
                                    <article className="tile is-child">
                                        <Scatterplot
                                            width={600}
                                            height={600}
                                            onClick={this.props.set_datum}
                                            onDragA={x => this.handleDragA(x)}
                                            onDragB={x => this.handleDragB(x)}
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
                                                    this.props.selected_datum
                                                }
                                                data={this.props.data}
                                                damage_boundary_a={
                                                    this.state.damage_boundary_a
                                                }
                                                damage_boundary_b={
                                                    this.state.damage_boundary_b
                                                }
                                            />
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div>
                            <h4 id="map" className="title">
                                Map
                            </h4>
                            <Map
                                data={this.props.data}
                                onClick={this.props.set_datum}
                                set_global_map={this.set_global_map}
                                damage_boundary_a={this.state.damage_boundary_a}
                                damage_boundary_b={this.state.damage_boundary_b}
                                selected_datum={this.props.selected_datum}
                                admin_regions={this.props.admin_regions}
                            />
                        </div>
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
