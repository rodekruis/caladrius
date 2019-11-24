import * as React from "react";
import * as jsPDF from "jspdf";
import * as leafletImage from "leaflet-image";

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
            document.querySelector(".table-global-container"),
            15,
            40,
            {
                width: 500,
                height: 500,
            }
        );
    }

    convert_leaflet_map_to_image(err, canvas) {
        this.setState({
            global_map_image: canvas.toDataURL("image/svg+xml", 1.0),
        });
    }

    add_addresses(doc, data) {
        doc.addPage();
        doc.setFontSize(HEADER_FONT_SIZE);
        doc.text("Addresses", 15, 20);
        doc.setFontSize(TEXT_FONT_SIZE);
        let address_table = [];
        data.map(datapoint => {
            address_table.push({
                damage: this.props.get_datum_priority(datapoint),
                address: datapoint.address || "ADDRESS NOT AVAILABLE",
            });
        });
        const address_table_header = [
            {
                name: "damage",
                prompt: "Damage",
                width: 40,
                align: "left",
                padding: 0,
            },
            {
                name: "address",
                prompt: "Address",
                width: 200,
                align: "left",
                padding: 0,
            },
        ];
        doc.table(15, 25, address_table, address_table_header);
    }

    create() {
        const doc = new jsPDF();
        this.add_title(doc);
        const timestamp = this.add_timestamp(doc);
        this.add_global_statistics(doc);
        this.add_addresses(doc, this.props.data);
        doc.save("Caladrius-Report-" + timestamp + ".pdf");
    }

    render() {
        return (
            <div className="report-container">
                <button
                    className="button is-primary report-button"
                    onClick={this.create}
                    disabled={!this.props.selected_model}
                >
                    Download Report
                </button>
            </div>
        );
    }
}
