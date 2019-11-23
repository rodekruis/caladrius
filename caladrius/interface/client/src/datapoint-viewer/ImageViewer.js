import * as React from "react";
import { DatumImage } from "./DatumImage";
import "./image_viewer.css";

export class ImageViewer extends React.Component {
    render() {
        return this.props.selected_datum ? (
            <div className="tile">
                <div className="tile is-parent">
                    <article className="tile is-child">
                        <h4 id="before-event-header" className="title is-4">
                            Before Event
                        </h4>
                        <div className="image-container before-image-container is-flex">
                            <DatumImage
                                image_label={"before"}
                                selected_datum={this.props.selected_datum}
                            />
                        </div>
                    </article>
                </div>
                <div className="tile is-parent">
                    <article className="tile is-child">
                        <h4 id="after-event-header" className="title is-4">
                            After Event
                        </h4>
                        <div className="image-container after-image-container is-flex">
                            <DatumImage
                                image_label={"after"}
                                selected_datum={this.props.selected_datum}
                            />
                        </div>
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
