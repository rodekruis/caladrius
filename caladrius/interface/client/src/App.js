import * as React from "react";
import { load_csv_data, load_admin_regions } from "./data.js";
import { ModelSelector } from "./nav/ModelSelector";
import { Nav } from "./nav/Nav";
import { Dashboard } from "./dashboard/Dashboard";
import { AddressList } from "./address-list/AddressList";
import { Footer } from "./footer/Footer";

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            models: [],
            selected_model: "",
            data: [],
            selected_datum: {},
            admin_regions: [],
        };
        this.load_model = this.load_model.bind(this);
        this.set_datum = this.set_datum.bind(this);
        this.render_model_selector = this.render_model_selector.bind(this);
    }

    componentDidMount() {
        load_admin_regions(admin_regions => {
            this.setState({ admin_regions: admin_regions });
        });
        fetch("/api/models")
            .then(res => res.json())
            .then(models => {
                this.setState({ models: models });
            });
    }

    load_model(model) {
        const model_name = model.model_directory;
        const prediction_filename = model.predictions.test[0];
        load_csv_data(model_name, prediction_filename, data => {
            this.setState({ selected_model: model, data: data });
        });
    }

    set_datum(datum) {
        this.setState({
            selected_datum: datum,
        });
    }

    render_model_selector() {
        return (
            <ModelSelector
                models={this.state.models}
                selected_model={this.state.selected_model}
                load_model={this.load_model}
            />
        );
    }

    render() {
        return (
            <div>
                <Nav
                    render_model_selector={this.render_model_selector}
                    data={this.state.data}
                />
                <Dashboard
                    data={this.state.data}
                    selected_datum={this.state.selected_datum}
                    admin_regions={this.state.admin_regions}
                    set_datum={this.set_datum}
                    render_model_selector={this.render_model_selector}
                    selected_model={this.state.selected_model}
                />
                <AddressList
                    data={this.state.data}
                    view_datapoint={this.set_datum}
                    selected_datum={this.state.selected_datum}
                />
                <Footer />
            </div>
        );
    }
}
