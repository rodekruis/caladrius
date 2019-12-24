import * as React from "react";
import { Auth } from "./auth/Auth";
import { Login } from "./auth/Login";
import { fetch_csv_data, fetch_admin_regions } from "./data.js";
import { Nav } from "./nav/Nav";
import { Breadcrumb } from "./breadcrumb/Breadcrumb";
import { ModelList } from "./model-list/ModelList";
import { Dashboard } from "./dashboard/Dashboard";
import { Footer } from "./footer/Footer";
import "./app.css";

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            is_authenticated: false,
            login_attempted: false,
            username: null,
            models: [],
            selected_model: null,
            data: [],
            selected_datum: null,
            admin_regions: [],
            loading: false,
            get_datum_priority: this.get_datum_priority_function(),
            selected_dataset_split: "test",
        };
    }

    componentDidMount() {
        Auth.auth(response => {
            this.setState(
                { is_authenticated: !!response, username: response },
                this.load_admin_regions_and_models
            );
        });
    }

    fetch_models = callback => {
        fetch("/api/models")
            .then(response => response.json())
            .then(callback);
    };

    load_admin_regions_and_models = () => {
        if (this.state.is_authenticated) {
            this.setState(
                { admin_regions: [], models: [], loading: true },
                () => {
                    fetch_admin_regions(admin_regions => {
                        this.fetch_models(models => {
                            if ("errno" in models) {
                                models = [];
                            }
                            this.setState({
                                admin_regions: admin_regions,
                                models: models,
                                loading: false,
                            });
                        });
                    });
                }
            );
        }
    };

    on_login = (username, password) => {
        const login_handler = success => {
            this.setState(
                {
                    is_authenticated: success,
                    login_attempted: !success,
                    username: username,
                    loading: false,
                },
                this.load_admin_regions_and_models
            );
        };

        this.setState({ login_attempted: true, loading: true }, () => {
            Auth.login(username, password, login_handler);
        });
    };

    unselect_model = () => {
        this.setState({ selected_model: null, selected_datum: null });
    };

    set_dataset_split = split_name => {
        this.setState({ selected_dataset_split: split_name }, () =>
            this.load_model(this.state.selected_model, split_name)
        );
    };

    on_exit = () => {
        if (this.state.selected_model) {
            this.unselect_model();
        } else {
            Auth.logout(() => {
                this.setState({
                    is_authenticated: false,
                    username: null,
                    login_attempted: false,
                });
            });
        }
    };

    load_model = (model, split = "test") => {
        this.setState(
            {
                selected_model: null,
                selected_dataset_split: split,
                selected_datum: null,
                data: [],
                loading: true,
            },
            () => {
                const model_name = model.model_directory;
                const prediction_filename =
                    this.state.selected_dataset_split === "test"
                        ? model.test_prediction_file_name
                        : model.validation_prediction_file_name.slice(-1)[0];
                fetch_csv_data(model_name, prediction_filename, data => {
                    this.setState({
                        selected_model: model,
                        selected_dataset_split: split,
                        selected_datum: null,
                        data: data,
                        loading: false,
                    });
                });
            }
        );
    };

    set_datum = datum => {
        this.setState(prevState => {
            return {
                selected_datum:
                    prevState.selected_datum === datum ? null : datum,
            };
        });
    };

    get_datum_priority_function = (lower_bound = 0.3, upper_bound = 0.7) => {
        return datum => {
            if (datum.prediction < lower_bound) {
                return "Low";
            } else if (datum.prediction > upper_bound) {
                return "High";
            } else {
                return "Medium";
            }
        };
    };

    set_datum_priority = (lower_bound, upper_bound) => {
        this.setState({
            get_datum_priority: this.get_datum_priority_function(
                lower_bound,
                upper_bound
            ),
        });
    };

    render_loader = () => {
        return (
            <section className="hero is-large">
                <div className="hero-body">
                    <div className="container">
                        <h2 className="title">
                            <progress className="progress is-warning" max="100">
                                100%
                            </progress>
                        </h2>
                    </div>
                </div>
            </section>
        );
    };

    render_nav = () => {
        return (
            <Nav
                data={this.state.data}
                selected_model={this.state.selected_model}
                get_datum_priority={this.state.get_datum_priority}
                loading={this.state.loading}
                on_exit={this.on_exit}
                is_authenticated={this.state.is_authenticated}
            />
        );
    };

    render_dashboard = () => {
        return (
            <Dashboard
                data={this.state.data}
                selected_datum={this.state.selected_datum}
                admin_regions={this.state.admin_regions}
                set_datum={this.set_datum}
                set_datum_priority={this.set_datum_priority}
                get_datum_priority={this.state.get_datum_priority}
            />
        );
    };

    render_model_list = () => {
        return (
            <ModelList
                models={this.state.models}
                load_model={this.load_model}
            />
        );
    };

    render_footer = () => {
        return <Footer />;
    };

    render_login = () => {
        return (
            <Login
                on_login={this.on_login}
                is_authenticated={this.state.is_authenticated}
                login_attempted={this.state.login_attempted}
            />
        );
    };

    render_breadcrumb = () => {
        return (
            <Breadcrumb
                loading={this.state.loading}
                selected_model={this.state.selected_model}
                unselect_model={this.unselect_model}
                selected_datum={this.state.selected_datum}
                set_datum={this.set_datum}
                selected_dataset_split={this.state.selected_dataset_split}
                set_dataset_split={this.set_dataset_split}
            />
        );
    };

    render() {
        return (
            <div>
                {this.render_nav(this.state.is_authenticated)}
                {this.render_breadcrumb()}
                {this.state.loading
                    ? this.render_loader()
                    : this.state.is_authenticated
                    ? this.state.selected_model
                        ? this.render_dashboard()
                        : this.render_model_list()
                    : this.render_login()}
                {this.render_footer()}
            </div>
        );
    }
}
