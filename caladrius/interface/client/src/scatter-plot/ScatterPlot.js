import React from "react";
import * as d3 from "d3";
import { Axes } from "./Axes";
import { Circles } from "./Circles";
import { DamageBoundary } from "./DamageBoundary";
import { yellow, red } from "../colours";
import "./scatter_plot.css";

export class ScatterPlot extends React.Component {
    constructor(props) {
        super(props);
        let range = [0, 100]; // svg components are aligned according to range [0, 100]
        let domain = [0, 1];
        let x_scale = d3
            .scaleLinear()
            .range(range)
            .domain(domain);
        let inverse_x_scale = d3
            .scaleLinear()
            .range(domain)
            .domain(range);
        let y_scale = d3
            .scaleLinear()
            .range(range.slice().reverse())
            .domain(domain);
        this.state = {
            axis_scales: {
                x_scale: x_scale,
                y_scale: y_scale,
                inverse_x_scale: inverse_x_scale,
            },
        };
    }

    render() {
        return (
            <div className="scatter-plot-container">
                <h4 id="map" className="title">
                    Correlation Graph
                </h4>
                <svg
                    height="100%"
                    width="100%"
                    viewBox="-10 -10 120 120"
                    preserveAspectRatio="xMinYMin meet"
                >
                    <g>
                        <Axes axis_scales={this.state.axis_scales} />
                        <Circles
                            data={this.props.data}
                            axis={this.state.axis_scales}
                            set_datum={this.props.set_datum}
                            selected_datum={this.props.selected_datum}
                            damage_boundary_a={this.props.damage_boundary_a}
                            damage_boundary_b={this.props.damage_boundary_b}
                        />
                        <DamageBoundary
                            axis={this.state.axis_scales}
                            onDrag={this.props.onDragA}
                            x={this.props.damage_boundary_a}
                            xmin={0.01}
                            xmax={this.props.damage_boundary_b - 0.01}
                            stroke={yellow}
                        />
                        <DamageBoundary
                            axis={this.state.axis_scales}
                            onDrag={this.props.onDragB}
                            x={this.props.damage_boundary_b}
                            xmin={this.props.damage_boundary_a + 0.01}
                            xmax={0.99}
                            stroke={red}
                        />
                    </g>
                </svg>
            </div>
        );
    }
}
