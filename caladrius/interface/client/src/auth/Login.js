import * as React from "react";

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
            <div className="login container">
                <form className="login-form" onSubmit={this.submit_handler}>
                    <div className="card login-card z-depth-5">
                        <div className="card-content">
                            <div className="card-title center-align">
                                <h4>Welcome</h4>
                            </div>
                            <div className="row">
                                <div className="input-field col s12">
                                    <input
                                        id="username"
                                        name="username"
                                        type="text"
                                        className="validate"
                                        value={this.state.username}
                                        onChange={this.input_change_handler}
                                        required={true}
                                        autoComplete="username"
                                    />
                                    <label htmlFor="username">Username</label>
                                </div>
                            </div>
                            <div className="row">
                                <div className="input-field col s12">
                                    <input
                                        id="password"
                                        name="password"
                                        type="password"
                                        className="validate"
                                        value={this.state.password}
                                        onChange={this.input_change_handler}
                                        required={true}
                                        autoComplete="current-password"
                                    />
                                    <label htmlFor="password">Password</label>
                                </div>
                            </div>
                            <div className="row">
                                <button
                                    type="submit"
                                    className="button is-primary"
                                >
                                    Enter
                                </button>
                            </div>
                        </div>
                    </div>
                </form>
            </div>
        );
    }
}
