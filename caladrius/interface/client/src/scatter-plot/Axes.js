import React from "react";
import * as d3 from "d3";

export class Axes extends React.Component {
    render() {
        return (
            <g>
                <Axis
                    scale={this.props.axis_scales.x_scale}
                    className="x-axis axis"
                    axisTransform="translate(0, 100)"
                    axis={d3.axisBottom}
                    labelTransform="translate(90, 99) rotate(0)"
                    labelText="Prediction"
                />
                <Axis
                    scale={this.props.axis_scales.y_scale}
                    className="y-axis axis"
                    axisTransform="translate(0, 0)"
                    axis={d3.axisLeft}
                    labelTransform="translate(5, 5) rotate(-90)"
                    labelText="Label"
                />
            </g>
        );
    }
}

export class Axis extends React.Component {
    constructor(props) {
        super(props);
        this.ref = React.createRef();
    }

    componentDidMount() {
        d3.select(this.ref.current)
            .call(this.props.axis(this.props.scale).tickSize(0))
            .append("text")
            .attr("class", `${this.props.className}-label`)
            .attr("transform", this.props.labelTransform)
            .text(this.props.labelText);
    }

    render() {
        return (
            <g
                className={this.props.className}
                transform={this.props.axisTransform}
                ref={this.ref}
            />
        );
    }
}
