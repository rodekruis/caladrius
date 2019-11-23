import * as React from "react";

export class Footer extends React.Component {
    render() {
        return (
            <footer className="footer">
                <div className="content has-text-centered">
                    <p>
                        <strong>Caladrius</strong> is built by{" "}
                        <a href="https://www.510.global/">510.global</a> under
                        the{" "}
                        <a href="https://github.com/rodekruis/caladrius/blob/master/LICENSE">
                            GPL-3.0 license
                        </a>
                        .
                    </p>
                </div>
            </footer>
        );
    }
}
