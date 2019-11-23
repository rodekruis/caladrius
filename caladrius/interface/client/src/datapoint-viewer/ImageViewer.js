import * as React from "react";
import { MapImage } from "./MapImage";

export class ImageViewer extends React.Component {
    render() {
        return (
            <div className="tile">
                <div className="tile is-parent is-vertical">
                    <article className="tile is-child">
                        <h4 id="map" className="title is-4">
                            Before Image
                        </h4>
                        <div className="map-before-image-container is-flex">
                            <MapImage
                                image_label={"before"}
                                selected_datum={this.props.selected_datum}
                            />
                        </div>
                    </article>
                </div>
                <div className="tile is-parent">
                    <article className="tile is-child">
                        <h4 id="map" className="title is-4">
                            After Image
                        </h4>
                        <div className="map-after-image-container is-flex">
                            <MapImage
                                image_label={"after"}
                                selected_datum={this.props.selected_datum}
                            />
                        </div>
                    </article>
                </div>
            </div>
        );
    }
}
