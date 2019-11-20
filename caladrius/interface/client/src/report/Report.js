import * as React from "react";
import * as jsPDF from "jspdf";
import * as leafletImage from "leaflet-image";
import "./report.css";

const TITLE_FONT_SIZE = 36;
const HEADER_FONT_SIZE = 24;
const TIMESTAMP_FONT_SIZE = 11;
const TEXT_FONT_SIZE = 14;

export class Report extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.create = this.create.bind(this);
    }

    componentDidMount() {
        console.log(this.props.globalMap);
    }

    add_title(doc) {
        doc.setFontSize(TITLE_FONT_SIZE);
        doc.text("Caladrius Report", 15, 25);
    }

    add_timestamp(doc) {
        const timestamp = new Date();
        const timestampISOFormat = timestamp.toISOString();
        const timestampReadableFormat = timestamp.toLocaleString();
        doc.setFontSize(TIMESTAMP_FONT_SIZE);
        doc.text("Created on: " + timestampReadableFormat, 17, 35);
        return timestampISOFormat;
    }

    add_global_statistics(doc) {
        doc.setFontSize(TEXT_FONT_SIZE);
        doc.fromHTML(
            document.querySelector(".table-selection-container"),
            15,
            40,
            {
                width: 10,
                height: 10,
            }
        );
    }

    convert_leaflet_map_to_image(err, canvas) {
        this.setState({ global_map_image: canvas.toDataURL("image/svg+xml", 1.0) });
    }

    add_global_map(doc, map) {
        // var dimensions = map.getSize();
        leafletImage(this.props.globalMap, this.convert_leaflet_map_to_image);
        console.log(this.state.global_map_image);
        doc.addImage(this.state.global_map_image, 'PNG', 10, 10, 100, 100);
    }

    add_addresses(doc) {
        doc.addPage();
        doc.setFontSize(HEADER_FONT_SIZE);
        doc.text("Addresses", 15, 20);
    }

    create() {
        const doc = new jsPDF();
        this.add_title(doc);
        const timestamp = this.add_timestamp(doc);
        this.add_global_statistics(doc);
        console.log(this.props.globalMap);
        this.add_global_map(doc, this.props.globalMap);
        this.add_addresses(doc);
        doc.save("Caladrius-Report-" + timestamp + ".pdf");
    }

    render() {
        return (
            <div className="report-container">
                <button className="report-button" onClick={this.create}>
                    Download Report
                </button>
            </div>
        );
    }
}
