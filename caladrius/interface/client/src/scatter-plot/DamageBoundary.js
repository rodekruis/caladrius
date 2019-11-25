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
            .subject(d => {
                return d3.select(this.ref);
            })
            .on("start", function() {
                d3.event.sourceEvent.stopPropagation();
                d3.select(this).classed("dragging", true);
            })
            .on("drag", function() {
                let x = d3.event.x;
                let xnew = x < xmin ? xmin : x > xmax ? xmax : x;
                d3.select(this)
                    .attr("x1", xnew)
                    .attr("x2", xnew);
            })
            .on("end", function() {
                d3.select(this).classed("dragging", false);
                that.props.onDrag(
                    that.props.axis.inverseXScale(d3.select(this).attr("x1"))
                );
            });

        if (this.ref) {
            d3.select(this.ref)
                .attr("x1", this.props.axis.xScale(this.props.x))
                .attr("x2", this.props.axis.xScale(this.props.x))
                .attr("y1", 0)
                .attr("y2", this.props.axis.height)
                .attr("stroke", this.props.stroke)
                .attr("stroke-width", 2)
                .call(drag);
        }
    }

    render() {
        return <line ref={ref => (this.ref = ref)} />;
    }
}
