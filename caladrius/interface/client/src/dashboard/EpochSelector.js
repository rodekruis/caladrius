import * as React from "react";

export class EpochSelector extends React.Component {
    render() {
        return (
            <nav
                className="pagination is-centered is-rounded"
                role="navigation"
                aria-label="pagination"
            >
                <a
                    href="#/"
                    className="pagination-previous"
                    onClick={() => {
                        this.props.pause_epoch();
                        this.props.set_epoch(Math.max(this.props.epoch - 1, 1));
                    }}
                    disabled={this.props.epoch === 1}
                >
                    Previous Epoch
                </a>
                <a
                    href="#/"
                    className="pagination-next"
                    onClick={() => {
                        this.props.pause_epoch();
                        this.props.set_epoch(
                            Math.min(
                                this.props.epoch + 1,
                                this.props.number_of_epochs
                            )
                        );
                    }}
                    disabled={this.props.epoch === this.props.number_of_epochs}
                >
                    Next Epoch
                </a>
                <ul className="pagination-list">
                    <li
                        className={this.props.epoch <= 2 ? "is-hidden" : ""}
                        onClick={() => {
                            this.props.pause_epoch();
                            this.props.set_epoch(1);
                        }}
                    >
                        <a
                            href="#/"
                            className="pagination-link"
                            aria-label="Goto epoch 1"
                        >
                            1
                        </a>
                    </li>
                    <li className={this.props.epoch <= 3 ? "is-hidden" : ""}>
                        <span className="pagination-ellipsis">&hellip;</span>
                    </li>
                    <li
                        className={this.props.epoch === 1 ? "is-hidden" : ""}
                        onClick={() => {
                            this.props.pause_epoch();
                            this.props.set_epoch(this.props.epoch - 1);
                        }}
                    >
                        <a
                            href="#/"
                            className="pagination-link"
                            aria-label={"Goto epoch " + (this.props.epoch - 1)}
                        >
                            {this.props.epoch - 1}
                        </a>
                    </li>
                    <li>
                        <a
                            href="#/"
                            className="pagination-link is-current"
                            onClick={
                                this.props.epoch_playing
                                    ? this.props.pause_epoch
                                    : this.props.play_epoch
                            }
                            aria-label={"Epoch " + this.props.epoch}
                            aria-current="page"
                        >
                            {this.props.epoch_playing ? ">" : this.props.epoch}
                        </a>
                    </li>
                    <li
                        className={
                            this.props.epoch === this.props.number_of_epochs
                                ? "is-hidden"
                                : ""
                        }
                        onClick={() => {
                            this.props.pause_epoch();
                            this.props.set_epoch(this.props.epoch + 1);
                        }}
                    >
                        <a
                            href="#/"
                            className="pagination-link"
                            aria-label={"Goto epoch " + (this.props.epoch + 1)}
                        >
                            {this.props.epoch + 1}
                        </a>
                    </li>
                    <li
                        className={
                            this.props.epoch >= this.props.number_of_epochs - 2
                                ? "is-hidden"
                                : ""
                        }
                    >
                        <span className="pagination-ellipsis">&hellip;</span>
                    </li>
                    <li
                        className={
                            this.props.epoch >= this.props.number_of_epochs - 1
                                ? "is-hidden"
                                : ""
                        }
                        onClick={() => {
                            this.props.pause_epoch();
                            this.props.set_epoch(this.props.number_of_epochs);
                        }}
                    >
                        <a
                            href="#/"
                            className="pagination-link"
                            aria-label={
                                "Goto epoch " + this.props.number_of_epochs
                            }
                        >
                            {this.props.number_of_epochs}
                        </a>
                    </li>
                </ul>
            </nav>
        );
    }
}
