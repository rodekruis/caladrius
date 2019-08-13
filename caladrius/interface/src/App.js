import * as React from "react";
import { load_csv_data, load_admin_regions } from './data.js'
import { Scatterplot } from "./Scatterplot";
import { MapImage } from "./MapImage";
import { PointInfoTable, CountAvgTable } from "./Tables";
import { Map } from "./Map"
import './app.css';

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [],
            admin_regions: [],
            selected_datum: {},
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7,
            map_center: [18.0425, -63.0548]
        };
    };

    componentDidMount() {
        const data = load_csv_data()
        data.then(d => this.setState({ data: d }))
        const admin_regions = load_admin_regions()
        admin_regions.then(d => this.setState({ admin_regions: d }))
    }

    handleClick(datum) {
        this.setState({
            selected_datum: datum
        })
    }

    handleDragA(x) {
        this.setState({ damage_boundary_a: x })
    }

    handleDragB(x) {
        this.setState({ damage_boundary_b: x })
    }

    render() {
        return (
            <div className="app-container">
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
                                image_label={'before'}
                                selected_datum={this.state.selected_datum}
                            />
                        </div>
                        <div className="map-after-image-container">
                            <MapImage
                                image_label={'after'}
                                selected_datum={this.state.selected_datum}
                            />
                        </div>
                    </div>
                    <div className="widget-container">
                        <div className="tables-container">
                            <div className="table-selection-container">
                                <PointInfoTable
                                    selected_datum={this.state.selected_datum}
                                    table_id={'infoToolTipBox'}
                                />
                            </div>
                            <div className="table-global-container">
                                <CountAvgTable
                                    data={this.state.data}
                                    damage_boundary_a={this.state.damage_boundary_a}
                                    damage_boundary_b={this.state.damage_boundary_b}
                                    table_id={'countAvgTable'}
                                />
                            </div>
                        </div>
                        <div className="map-container">
                            <Map
                                width={400}
                                height={404}
                                data={this.state.data}
                                onClick={datum => this.handleClick(datum)}
                                damage_boundary_a={this.state.damage_boundary_a}
                                damage_boundary_b={this.state.damage_boundary_b}
                                selected_datum={this.state.selected_datum}
                                admin_regions={this.state.admin_regions}
                            />
                        </div>
                    </div>
                </div>
            </div>
        )
    };
}

