import * as React from "react";
import * as jsPDF from "jspdf";
import "./report.css";

const TITLE_FONT_SIZE = 36;
const HEADER_FONT_SIZE = 24;
const TIMESTAMP_FONT_SIZE = 11;

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

    add_addresses(doc) {
        doc.addPage();
        doc.setFontSize(HEADER_FONT_SIZE);
        doc.text("Addresses", 15, 20);
    }

    create() {
        const doc = new jsPDF();
        this.add_title(doc);
        const timestamp = this.add_timestamp(doc);
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
