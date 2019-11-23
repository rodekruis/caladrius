import React from "react";

export class DatumImage extends React.Component {
    constructor(props) {
        super(props);
        this.selectImage = this.selectImage.bind(this);
    }

    selectImage() {
        let image_key = "/510-logo.png";
        if (this.props.selected_datum) {
            image_key =
                "/" +
                this.props.image_label +
                "/" +
                this.props.selected_datum.filename;
        }
        return image_key;
    }

    render() {
        let image = this.selectImage();
        return (
            <img
                className="datum-image"
                src={image}
                alt={this.props.image_label}
            />
        );
    }
}
