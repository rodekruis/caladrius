import React from "react";
import * as d3 from "d3";
import {
    AxisBottom,
    AxisBottomLabel,
    AxisLeft,
    AxisLeftLabel,
    Title,
} from "./Axes";
import { Circles } from "./Circles";
import { DamageBoundary } from "./DamageBoundary";
import { least, heavy } from "../colours";

const margin = { top: 50, right: 50, bottom: 50, left: 50 };

export class Scatterplot extends React.Component {
    render() {
        let width = this.props.width - margin.left - margin.right;
        let height = this.props.height - margin.top - margin.bottom;
        let xScale = d3
            .scaleLinear()
            .range([0, width])
            .domain([0, 1]);
        let inverseXScale = d3
            .scaleLinear()
            .domain([0, width])
            .range([0, 1]);
        let yScale = d3
            .scaleLinear()
            .range([height, 0])
            .domain([0, 1]);
        let axis_props = {
            width: width,
            height: height,
            xScale: xScale,
            yScale: yScale,
            inverseXScale: inverseXScale,
            margin: margin,
        };

        return (
            <div>
                <h4 id="map" className="title is-4">
                    Correlation Graph
                </h4>
                <svg height={this.props.height} width={this.props.width}>
                    <g transform={`translate(${margin.left},${margin.top})`}>
                        <AxisBottom {...axis_props} />
                        <AxisLeft {...axis_props} />
                        <AxisBottomLabel {...axis_props} />
                        <AxisLeftLabel {...axis_props} />
                        <Title {...axis_props} />
                        <Circles
                            data={this.props.data}
                            axis={axis_props}
                            onClick={this.props.onClick}
                            selected_datum={this.props.selected_datum}
                            damage_boundary_a={this.props.damage_boundary_a}
                            damage_boundary_b={this.props.damage_boundary_b}
                        />
                        <DamageBoundary
                            axis={axis_props}
                            onDrag={this.props.onDragA}
                            x={this.props.damage_boundary_a}
                            xmin={0.0}
                            xmax={this.props.damage_boundary_b}
                            stroke={least}
                        />
                        <DamageBoundary
                            axis={axis_props}
                            onDrag={this.props.onDragB}
                            x={this.props.damage_boundary_b}
                            xmin={this.props.damage_boundary_a}
                            xmax={1.0}
                            stroke={heavy}
                        />
                    </g>
                </svg>
            </div>
        );
    }
}
