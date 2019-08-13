import React from 'react';
import { images, image_placeholder } from './images.js'
import './map_image.css';

export class MapImage extends React.Component {

    constructor(props) {
        super(props);
        this.selectImage = this.selectImage.bind(this)
    }

    selectImage() {
        if (Object.keys(this.props.selected_datum).length > 0) {
            let image_key = this.props.image_label + '/' + this.props.selected_datum.filename
            return images[image_key]
        }
        else {
            return image_placeholder
        }
    }

    render() {
        let image = this.selectImage()
        return (
            <img
                className='map-image'
                src={image}
                alt={this.props.image_label}
            />
        );
    }
}
