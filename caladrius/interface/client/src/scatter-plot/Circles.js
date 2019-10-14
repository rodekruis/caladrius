import React from 'react';
import * as d3 from 'd3';
import { get_point_colour, selected } from '../colours';


export class Circles extends React.Component {
    render() {
        const circles = this.props.data.map(datum => {
            return (
                <Circle
                    key={datum.objectId}
                    axis={this.props.axis}
                    datum={datum}
                    onClick={() => this.props.onClick(datum)}
                    selected_datum={this.props.selected_datum}
                    damage_boundary_a={this.props.damage_boundary_a}
                    damage_boundary_b={this.props.damage_boundary_b}
                />)
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
        this.draw_plot()
    }

    componentDidUpdate() {
        this.draw_plot()
    }

    draw_plot() {
        let that = this;
        d3.select(this.ref.current)
            .attr('cx', this.props.axis.xScale(this.props.datum.prediction))
            .attr('cy', this.props.axis.yScale(this.props.datum.label))
            .attr('r', 5)
            .attr('fill', () => get_point_colour(
                this.props.datum.prediction,
                this.props.damage_boundary_a,
                this.props.damage_boundary_b,
                this.props.datum.objectId,
                this.props.selected_datum.objectId
            ))
            .on('mouseover', function(d) {
                d3.select(this).attr('fill', selected);
            })
            .on('mouseout', function(d) {
                d3.select(this).attr('fill', () => get_point_colour(
                    that.props.datum.prediction,
                    that.props.damage_boundary_a,
                    that.props.damage_boundary_b,
                    that.props.datum.objectId,
                    that.props.selected_datum.objectId
                ));
            });
    }

    render() {
        return (
            <circle
                ref={this.ref}
                onClick={this.props.onClick}
            />
        )
    }
}
