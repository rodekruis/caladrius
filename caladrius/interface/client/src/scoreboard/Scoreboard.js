import * as React from "react";
import { BuildingStatsTable, ClassificationStatsTable } from "./Tables";
import "./scoreboard.css";

export class Scoreboard extends React.Component {
    render() {
        return (
            <div className="tile">
                <div className="tile is-parent">
                    <article className="tile is-child table-selection-container">
                        <h4 id="map" className="title is-4">
                            Building Stats
                        </h4>
                        {this.props.selected_datum ? (
                            <BuildingStatsTable
                                selected_datum={this.props.selected_datum}
                                table_id={"building-stats-table"}
                                get_datum_priority={
                                    this.props.get_datum_priority
                                }
                            />
                        ) : (
                            <div className="notification damage-stats-notification">
                                Click on a datapoint in the Correlation Graph to
                                view the building damage.
                            </div>
                        )}
                    </article>
                </div>
                <div className="tile is-parent">
                    <article className="tile is-child table-global-container">
                        <h4 id="map" className="title is-4">
                            Classification Stats
                        </h4>
                        <ClassificationStatsTable
                            data={this.props.data}
                            damage_boundary_a={this.props.damage_boundary_a}
                            damage_boundary_b={this.props.damage_boundary_b}
                            table_id={"classification-stats-table"}
                        />
                    </article>
                </div>
            </div>
        );
    }
}
