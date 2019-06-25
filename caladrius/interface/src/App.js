import * as React from "react";
import { load_csv_data } from './data.js'
import { Scatterplot } from "./Scatterplot";
import { MapImage } from "./MapImage";

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [],
            selected_datum_id: -1,
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7
        };
    };
  
  componentDidMount() {
    const data = load_csv_data()
    data.then(d => this.setState({data: d}))
  }

  handleClick(datum) {
    this.setState({selected_datum_id: datum.objectId}, function () {
      console.log(this.state.selected_datum_id) })
  }

  handleDragA(x) {
    this.setState({damage_boundary_a: x}, function () {
      console.log(this.state.damage_boundary_a) })
  }

  handleDragB(x) {
    this.setState({damage_boundary_b: x}, function () {
      console.log(this.state.damage_boundary_b) })
  }

  render() {
    return (
      <div>
        <div style={{ height: "300px", width: "100%" }}>
          <Scatterplot
           width={300}
           height={300}
           onClick={datum => this.handleClick(datum)}
           onDragA={x => this.handleDragA(x)}
           onDragB={x => this.handleDragB(x)}
           data={this.state.data}
           selected_datum_id={this.state.selected_datum_id}
           damage_boundary_a={this.state.damage_boundary_a}
           damage_boundary_b={this.state.damage_boundary_b}
          />
        </div>
        <div style={{ height: "200px", width: "200px" }}>
          <MapImage
           width={200}
           height={200}
           data={this.state.data}
           image_label={'before'}
           selected_datum_id={this.state.selected_datum_id}
          />
        </div>
        <div style={{ height: "200px", width: "200px" }}>
          <MapImage
           width={200}
           height={200}
           data={this.state.data}
           image_label={'after'}
           selected_datum_id={this.state.selected_datum_id}
          />
        </div>
      </div>
    )
  };
}

