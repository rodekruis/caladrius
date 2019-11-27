import React from "react";

export class DatumImage extends React.Component {
    constructor(props) {
        super(props);
        this.state = { switching_datum: true };
    }

    componentDidUpdate(prevProps, prevState) {
        if (prevProps.selected_datum !== this.props.selected_datum) {
            this.setState({ switching_datum: true });
        }
    }

    select_image() {
        let image_key = "/510-logo.png";
        if (this.props.selected_datum) {
            image_key =
                "/" +
                this.props.image_folder +
                "/" +
                this.props.selected_datum.filename;
        }
        return image_key;
    }

    render() {
        return (
            <div
                className={
                    "image-container is-flex" +
                    (this.state.switching_datum ? " image-loading" : "")
                }
            >
                <img
                    className="datum-image"
                    src={this.select_image()}
                    alt={this.props.image_label}
                    onLoad={() => {
                        this.setState({
                            switching_datum: false,
                        });
                    }}
                />
            </div>
        );
    }
}
