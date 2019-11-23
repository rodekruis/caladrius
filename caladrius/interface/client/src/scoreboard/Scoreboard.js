import * as React from "react";
import { PointInfoTable, CountAvgTable } from "./Tables";

export class Scoreboard extends React.Component {
    constructor(props) {
        super(props);
    }

    render() {
        return (
            <div className="tile is-parent">
                <article className="tile is-child table-selection-container">
                    <h4 id="map" className="title is-4">
                        Selected Datapoint
                    </h4>
                    <PointInfoTable
                        selected_datum={this.props.selected_datum}
                        table_id={"infoToolTipBox"}
                    />
                </article>
                <article className="tile is-child table-global-container">
                    <h4 id="map" className="title is-4">
                        Overall Stats
                    </h4>
                    <CountAvgTable
                        data={this.props.data}
                        damage_boundary_a={this.props.damage_boundary_a}
                        damage_boundary_b={this.props.damage_boundary_b}
                        table_id={"countAvgTable"}
                    />
                </article>
            </div>
        );
    }
}
