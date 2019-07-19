import React from 'react';
import { images, image_placeholder} from './images.js'

export class MapImage extends React.Component{

    constructor(props) {
      super(props);
      this.selectImage = this.selectImage.bind(this)
    }

    selectImage() {
        if (this.props.selected_datum_id > 0) {
            let datum = this.props.data.filter(d => 
               d.objectId === this.props.selected_datum_id)[0]
            let image_key = this.props.image_label + '/' + datum.filename 
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
           src={image}
           alt={this.props.image_label}
           width={this.props.width}
          />
      );
    }
}