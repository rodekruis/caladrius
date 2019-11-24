import React from "react";
import * as d3 from "d3";

class AxisComponent extends React.Component {
    constructor(props) {
        super(props);
        this.ref = React.createRef();
        this.createComponent = this.createComponent.bind(this);
    }

    componentDidMount() {
        if (this.ref.current) {
            this.createComponent();
        }
    }

    componentDidUpdate() {
        if (this.ref.current) {
            this.createComponent();
        }
    }

    creatComponent() {
        return;
    }

    render() {
        return;
    }
}

export class AxisBottom extends AxisComponent {
    createComponent() {
        d3.select(this.ref.current).call(d3.axisBottom(this.props.xScale));
    }

    render() {
        return (
            <g
                transform={`translate(0, ${this.props.height})`}
                ref={this.ref}
            />
        );
    }
}

export class AxisLeft extends AxisComponent {
    createComponent() {
        d3.select(this.ref.current).call(d3.axisLeft(this.props.yScale));
    }

    render() {
        return <g ref={this.ref} />;
    }
}

export class AxisBottomLabel extends AxisComponent {
    createComponent() {
        d3.select(this.ref.current)
            .attr("id", "xAxisLabel")
            .style("text-anchor", "middle")
            .text("Predicted");
    }

    render() {
        let x_trans = this.props.width / 2;
        let y_trans = this.props.height + this.props.margin.top / 1.5;
        return (
            <text
                transform={`translate(${x_trans}, ${y_trans})`}
                ref={this.ref}
            />
        );
    }
}

export class AxisLeftLabel extends AxisComponent {
    createComponent() {
        d3.select(this.ref.current)
            .attr("id", "yAxisLabel")
            .style("text-anchor", "middle")
            .text("Actual");
    }

    render() {
        let xpos = 0 - this.props.height / 2;
        let ypos = 0 - this.props.margin.left / 1.5;
        return (
            <text transform={`rotate(-90)`} x={xpos} y={ypos} ref={this.ref} />
        );
    }
}
