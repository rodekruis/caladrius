import * as React from "react";
import { DatumImage } from "./DatumImage";
import "./image_viewer.css";

export class ImageViewer extends React.Component {
    render() {
        return this.props.selected_datum ? (
            <div className="tile image-viewer-container">
                <div className="tile is-parent">
                    <article className="tile is-child">
                        <h4 id="before-event-header" className="title is-4">
                            Before Event
                        </h4>
                        <DatumImage
                            image_folder={"before"}
                            image_label={"Image Before Event"}
                            selected_datum={this.props.selected_datum}
                        />
                    </article>
                </div>
                <div className="tile is-parent">
                    <article className="tile is-child">
                        <h4 id="after-event-header" className="title is-4">
                            After Event
                        </h4>
                        <DatumImage
                            image_folder={"after"}
                            image_label={"Image After Event"}
                            selected_datum={this.props.selected_datum}
                        />
                    </article>
                </div>
            </div>
        ) : (
            <div className="notification image-viewer-notification">
                Click on a datapoint in the Correlation Graph to view the
                building damage.
            </div>
        );
    }
}
