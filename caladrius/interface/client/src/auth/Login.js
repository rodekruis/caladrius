import * as React from "react";
import "./login.css";

export class Login extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            username: "",
            password: "",
        };
    }

    input_change_handler = event => {
        const target = event.target;
        const value = target.value;
        const name = target.name;

        this.setState({
            [name]: value,
        });
    };

    submit_handler = event => {
        this.props.login(this.state.username, this.state.password);
        event.preventDefault();
    };

    render() {
        return (
            <div className="columns is-centered is-vcentered login-container">
                <div className="column is-narrow">
                    <article className="panel is-primary">
                        <p className="panel-heading">Welcome to Caladrius</p>
                        <div className="panel-block">
                            <form
                                className="login-form"
                                onSubmit={this.submit_handler}
                            >
                                <div className="field">
                                    <label className="label" htmlFor="username">
                                        Username
                                    </label>
                                    <div className="control">
                                        <input
                                            className="input"
                                            id="username"
                                            name="username"
                                            type="text"
                                            placeholder="demo"
                                            value={this.state.username}
                                            onChange={this.input_change_handler}
                                            required={true}
                                            autoComplete="username"
                                        />
                                    </div>
                                </div>
                                <div className="field">
                                    <label className="label" htmlFor="password">
                                        Password
                                    </label>
                                    <div className="control">
                                        <input
                                            className="input"
                                            id="password"
                                            name="password"
                                            type="password"
                                            placeholder="510.global"
                                            value={this.state.password}
                                            onChange={this.input_change_handler}
                                            required={true}
                                            autoComplete="current-password"
                                        />
                                    </div>
                                </div>
                                <div className="field">
                                    <div className="control">
                                        <label className="checkbox">
                                            <input type="checkbox" /> I agree to
                                            the{" "}
                                            <a href="#">terms and conditions</a>
                                            .
                                        </label>
                                    </div>
                                </div>
                                <div className="field is-grouped is-grouped-centered">
                                    <div className="control">
                                        <button
                                            className="button is-primary"
                                            type="submit"
                                        >
                                            Enter
                                        </button>
                                    </div>
                                </div>
                            </form>
                        </div>
                    </article>
                </div>
            </div>
        );
    }
}
