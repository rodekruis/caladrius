import React from 'react';
import * as d3 from 'd3';

export class AxisBottom extends React.Component{

 constructor(props) {
    super(props);
    this.ref = React.createRef();
  }

   componentDidMount() {
    if (this.ref.current) {
      d3.select(this.ref.current).call(d3.axisBottom(this.props.xScale));
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      d3.select(this.ref.current)
        .transition()
        .call(d3.axisBottom(this.props.xScale));
    }
  }

  render() {
    return <g transform={`translate(0, ${this.props.height})`} ref={this.ref} />;
  }

}

export class AxisLeft extends React.Component{

 constructor(props) {
    super(props);
    this.ref = React.createRef();
  }

   componentDidMount() {
    if (this.ref.current) {
      d3.select(this.ref.current).call(d3.axisLeft(this.props.yScale));
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      d3.select(this.ref.current)
        .transition()
        .call(d3.axisLeft(this.props.yScale));
    }
  }

  render() {
    return <g ref={this.ref} />;
  }

}
