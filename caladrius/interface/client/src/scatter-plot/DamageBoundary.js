import * as React from "react";
import * as d3 from "d3";

export class DamageBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.draw_plot = this.draw_plot.bind(this);
    }

    componentDidMount() {
        this.draw_plot();
    }

    componentDidUpdate() {
        this.draw_plot();
    }

    draw_plot() {
        let that = this;
        let xmax = this.props.axis.xScale(this.props.xmax);
        let xmin = this.props.axis.xScale(this.props.xmin);
        const drag = d3
            .drag()
            .on("drag", function() {
                let x = d3.event.x;
                let xnew = x < xmin ? xmin : x > xmax ? xmax : x;
                d3.select(this)
                    .select("line")
                    .attr("x1", xnew)
                    .attr("x2", xnew);
                d3.select(this)
                    .select("circle")
                    .attr("cx", xnew);
            })
            .on("end", function() {
                that.props.onDrag(
                    that.props.axis.inverseXScale(
                        d3
                            .select(this)
                            .select("circle")
                            .attr("cx")
                    )
                );
            });
        d3.select(this.ref).call(drag);
        d3.select(this.ref)
            .select("line")
            .style("cursor", "pointer")
            .on("mouseover", e => {
                d3.select(this.ref)
                    .select("circle")
                    .attr("r", 5);
            })
            .on("mouseout", e => {
                d3.select(this.ref)
                    .select("circle")
                    .attr("r", 1);
            });
    }

    render() {
        return (
            <g ref={ref => (this.ref = ref)}>
                <line
                    stroke={this.props.stroke}
                    strokeWidth={2}
                    x1={this.props.axis.xScale(this.props.x)}
                    x2={this.props.axis.xScale(this.props.x)}
                    y1={0}
                    y2={this.props.axis.height}
                />
                <circle
                    stroke={this.props.stroke}
                    strokeWidth={2}
                    fill={this.props.stroke}
                    r={1}
                    cx={this.props.axis.xScale(this.props.x)}
                    cy={this.props.axis.height / 25}
                    fillOpacity={1.0}
                />
            </g>
        );
    }
}
