import React from "react";
import * as d3 from "d3";
import { get_prediction_colour } from "../colours";

export class Circles extends React.Component {
    render() {
        const circles = this.props.data.map(datum => {
            return (
                <Circle
                    key={datum.object_id}
                    axis={this.props.axis}
                    datum={datum}
                    set_datum={() => this.props.set_datum(datum)}
                    selected_datum={this.props.selected_datum}
                    damage_boundary_a={this.props.damage_boundary_a}
                    damage_boundary_b={this.props.damage_boundary_b}
                />
            );
        });
        return <g className="circles">{circles}</g>;
    }
}

class Circle extends React.Component {
    constructor(props) {
        super(props);
        this.ref = React.createRef();
        this.draw_plot = this.draw_plot.bind(this);
    }

    componentDidMount() {
        this.draw_plot();
    }

    componentDidUpdate() {
        this.draw_plot();
    }

    draw_plot() {
        const maximum_opacity = 1.0;
        const minimum_opacity =
            this.props.selected_datum == this.props.datum ? 1.0 : 0.3;
        let that = this;
        d3.select(this.ref.current)
            .attr("cx", this.props.axis.xScale(this.props.datum.prediction))
            .attr("cy", this.props.axis.yScale(this.props.datum.label))
            .attr("r", 5)
            .attr("fill", () =>
                get_prediction_colour(
                    this.props.datum.prediction,
                    this.props.damage_boundary_a,
                    this.props.damage_boundary_b
                )
            )
            .attr("fill-opacity", minimum_opacity)
            .on("mouseover", function(d) {
                d3.select(this).attr("fill-opacity", maximum_opacity);
            })
            .on("mouseout", function(d) {
                d3.select(this).attr("fill-opacity", minimum_opacity);
            });
    }

    render() {
        return <circle ref={this.ref} onClick={this.props.set_datum} />;
    }
}
