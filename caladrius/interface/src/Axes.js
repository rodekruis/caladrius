import React from 'react';
import * as d3 from 'd3';

export class AxisBottom extends React.Component{
 constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.createBottomAxis = this.createBottomAxis.bind(this)
  }

   componentDidMount() {
    if (this.ref.current) {
      this.createBottomAxis()
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      this.createBottomAxis()
    }
  }

  createBottomAxis() {
      d3.select(this.ref.current).call(d3.axisBottom(this.props.xScale));
  }

  render() {
    return <g transform={`translate(0, ${this.props.height})`} ref={this.ref} />;
  }
}



export class AxisLeft extends React.Component{
 constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.createLeftAxis = this.createLeftAxis.bind(this)
  }

   componentDidMount() {
    if (this.ref.current) {
      this.createLeftAxis()
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      this.createLeftAxis()
    }
  }

  createLeftAxis() {
      d3.select(this.ref.current).call(d3.axisLeft(this.props.yScale));
  }

  render() {
    return <g ref={this.ref} />;
  }
}

export class AxisBottomLabel extends React.Component{
 constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.createBottomAxisLabel = this.createBottomAxisLabel.bind(this)
  }

   componentDidMount() {
    if (this.ref.current) {
      this.createBottomAxisLabel()
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      this.createBottomAxisLabel()
    }
  }

  createBottomAxisLabel() {
      d3.select(this.ref.current)
        .attr('id', 'xAxisLabel')
        .style('text-anchor', 'middle')
        .text('Predicted');
  }

  render() {
    let x_trans = this.props.width / 2
    let y_trans = this.props.height + this.props.margin.top / 1.5
    return (
      <text 
      transform={`translate(${x_trans}, ${y_trans})`} 
      ref={this.ref} 
      />
    );
  }
}

export class AxisLeftLabel extends React.Component{
 constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.createLeftAxisLabel = this.createLeftAxisLabel.bind(this)
  }

   componentDidMount() {
    if (this.ref.current) {
      this.createLeftAxisLabel()
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      this.createLeftAxisLabel()
    }
  }

  createLeftAxisLabel() {
      d3.select(this.ref.current)
        .attr('id', 'yAxisLabel')
        .style('text-anchor', 'middle')
        .text('Actual')
  }

  render() {
    let xpos = 0 - this.props.height / 2
    let ypos = 0 - this.props.margin.left / 1.5 
    console.log(ypos)
    return (
      <text 
      transform={`rotate(-90)`} 
      x={xpos}
      y={ypos}
      ref={this.ref} 
      />
    );
  }
}

export class Title extends React.Component{
 constructor(props) {
    super(props);
    this.ref = React.createRef();
    this.createTitle = this.createTitle.bind(this)
  }

   componentDidMount() {
    if (this.ref.current) {
      this.createTitle()
    }
  }

  componentDidUpdate() {
    if (this.ref.current) {
      this.createTitle()
    }
  }

  createTitle() {
      d3.select(this.ref.current)
        .attr('id', 'title')
        .style('text-anchor', 'middle')
        .text('Siamese Network Model')
  }

  render() {
    let x_trans = this.props.width / 2
    let y_trans = 0 - this.props.margin.top / 2.0
    return (
      <text 
      transform={`translate(${x_trans}, ${y_trans})`} 
      ref={this.ref} 
      />
    )
  }
}

