import * as React from "react";
import { load_csv_data } from './data.js'
import { Scatterplot } from "./Scatterplot";
import { MapImage } from "./MapImage";
import { PointInfoTable, CountAvgTable } from "./Tables";
import { Map } from "./Map"

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: [],
            selected_datum: {},
            damage_boundary_a: 0.3,
            damage_boundary_b: 0.7,
            map_center: [18.0425, -63.0548]
        };
    };
  
  componentDidMount() {
    const data = load_csv_data()
    data.then(d => this.setState({data: d}))
  }

  handleClick(datum) {
    this.setState({
      selected_datum: datum
    })
  }

  handleDragA(x) {
    this.setState({damage_boundary_a: x})
  }

  handleDragB(x) {
    this.setState({damage_boundary_b: x})
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
           selected_datum={this.state.selected_datum}
           damage_boundary_a={this.state.damage_boundary_a}
           damage_boundary_b={this.state.damage_boundary_b}
          />
        </div>
        <div style={{ height: "200px", width: "200px" }}>
          <MapImage
           width={200}
           height={200}
           image_label={'before'}
           selected_datum={this.state.selected_datum}
          />
        </div>
        <div style={{ height: "200px", width: "200px" }}>
          <MapImage
           width={200}
           height={200}
           image_label={'after'}
           selected_datum={this.state.selected_datum}
          />
        </div>
       <div style={{width: "200px" }}>
          <PointInfoTable
           selected_datum={this.state.selected_datum}
           table_id={'infoToolTipBox'}
          />
        </div>
        <div style={{width: "200px" }}>
          <CountAvgTable
           data={this.state.data}
           damage_boundary_a={this.state.damage_boundary_a}
           damage_boundary_b={this.state.damage_boundary_b}
           table_id={'countAvgTable'}
          />
        </div>
        <div style={{width: "400px", height: "400px"}}>
          <Map
           width={400}
           height={400}
           data={this.state.data}
           onClick={datum => this.handleClick(datum)}
           damage_boundary_a={this.state.damage_boundary_a}
           damage_boundary_b={this.state.damage_boundary_b}
           selected_datum={this.state.selected_datum}
          />
        </div>
      </div>
    )
  };
}

