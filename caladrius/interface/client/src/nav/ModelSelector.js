import * as React from "react";

export class ModelSelector extends React.Component {
    constructor(props) {
        super(props);
        this.load_model = this.load_model.bind(this);
    }

    load_model(event) {
        const selected_model = this.props.models[event.target.value];
        this.props.load_model(selected_model);
        event.preventDefault();
    }

    show_models() {
        let items = [
            <option key={""} value={""} disabled>
                Choose a trained model
            </option>,
        ];
        this.props.models.forEach((model, index) => {
            items.push(
                <option key={model.model_name} value={index}>
                    {model.model_name}
                </option>
            );
        });
        return items;
    }

    render() {
        return (
            <div className="select">
                <select
                    value={
                        this.props.selected_model
                            ? this.props.models.indexOf(
                                  this.props.selected_model
                              )
                            : ""
                    }
                    onChange={this.load_model}
                >
                    {this.show_models()}
                </select>
            </div>
        );
    }
}
