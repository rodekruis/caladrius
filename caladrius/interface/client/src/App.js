import * as React from "react";
import { Auth } from "./auth/Auth";
import { Login } from "./auth/Login";
import { fetch_csv_data, fetch_admin_regions } from "./data.js";
import { ModelSelector } from "./nav/ModelSelector";
import { Nav } from "./nav/Nav";
import { Dashboard } from "./dashboard/Dashboard";
import { Footer } from "./footer/Footer";

export class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            is_authenticated: false,
            login_attempted: false,
            username: null,
            models: [],
            selected_model: "",
            data: [],
            selected_datum: null,
            admin_regions: [],
            loading: false,
            get_datum_priority: this.get_datum_priority_function(),
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
            this.setState({ loading: true }, () => {
                fetch_admin_regions(admin_regions => {
                    this.fetch_models(models => {
                        this.setState({
                            admin_regions: admin_regions,
                            models: models,
                            loading: false,
                        });
                    });
                });
            });
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

        this.setState({ loading: true, login_attempted: true }, () => {
            Auth.login(username, password, login_handler);
        });
    };

    on_logout = () => {
        Auth.logout(() => {
            this.setState({ is_authenticated: false, username: null });
        });
    };

    load_model = model => {
        this.setState({ loading: true }, () => {
            const model_name = model.model_directory;
            const prediction_filename = model.predictions.test[0];
            fetch_csv_data(model_name, prediction_filename, data => {
                this.setState({
                    selected_model: model,
                    data: data,
                    loading: false,
                });
            });
        });
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

    render_model_selector = () => {
        return (
            <ModelSelector
                models={this.state.models}
                selected_model={this.state.selected_model}
                load_model={this.load_model}
                loading={this.state.loading}
            />
        );
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
                render_model_selector={this.render_model_selector}
                data={this.state.data}
                selected_model={this.state.selected_model}
                get_datum_priority={this.state.get_datum_priority}
                loading={this.state.loading}
                on_logout={this.on_logout}
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
                render_model_selector={this.render_model_selector}
                selected_model={this.state.selected_model}
                set_datum_priority={this.set_datum_priority}
                get_datum_priority={this.state.get_datum_priority}
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

    render() {
        return (
            <div>
                {this.render_nav(this.state.is_authenticated)}
                {this.state.loading
                    ? this.render_loader()
                    : this.state.is_authenticated
                    ? this.render_dashboard()
                    : this.render_login()}
                {this.render_footer()}
            </div>
        );
    }
}
