import * as React from "react";
import * as d3 from "d3";

export class DamageBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.ref = React.createRef();
        this.draw = this.draw.bind(this);
        this.drag_handler = this.drag_handler.bind(this);
        this.drag_end_handler = this.drag_end_handler.bind(this);
    }

    componentDidMount() {
        this.draw();
    }

    componentDidUpdate() {
        this.draw();
    }

    drag_handler() {
        const xmax = this.props.axis.x_scale(this.props.xmax);
        const xmin = this.props.axis.x_scale(this.props.xmin);
        const x = d3.event.x;
        const xnew = x < xmin ? xmin : x > xmax ? xmax : x;
        d3.select(this.ref.current)
            .select("line")
            .attr("x1", xnew)
            .attr("x2", xnew);
        d3.select(this.ref.current)
            .select("circle")
            .attr("cx", xnew);
    }

    drag_end_handler() {
        this.props.onDrag(
            this.props.axis.inverse_x_scale(
                d3
                    .select(this.ref.current)
                    .select("circle")
                    .attr("cx")
            )
        );
    }

    draw() {
        const drag = d3
            .drag()
            .on("drag", this.drag_handler)
            .on("end", this.drag_end_handler);
        d3.select(this.ref.current)
            .call(drag)
            .select("line")
            .on("mouseover", () => {
                d3.select(this.ref.current)
                    .select("circle")
                    .attr("r", 2)
                    .attr("cy", d3.mouse(this.ref.current)[1]);
            })
            .on("mouseout", () => {
                d3.select(this.ref.current)
                    .select("circle")
                    .attr("r", 0)
                    .attr("cy", d3.mouse(this.ref.current)[1]);
            });
    }

    render() {
        return (
            <g ref={this.ref} className="damage-boundary">
                <circle
                    stroke={this.props.stroke}
                    strokeWidth={1}
                    fill={this.props.stroke}
                    fillOpacity={1.0}
                    r={2}
                    cx={this.props.axis.x_scale(this.props.x)}
                    cy={50}
                />
                <line
                    stroke={this.props.stroke}
                    strokeWidth={1}
                    x1={this.props.axis.x_scale(this.props.x)}
                    x2={this.props.axis.x_scale(this.props.x)}
                    y1={-1}
                    y2={102}
                />
                <line
                    stroke={this.props.stroke}
                    strokeWidth={1}
                    x1={-1}
                    x2={102}
                    y1={this.props.axis.x_scale(1.0 - this.props.x)}
                    y2={this.props.axis.x_scale(1.0 - this.props.x)}
                />
            </g>
        );
    }
}
