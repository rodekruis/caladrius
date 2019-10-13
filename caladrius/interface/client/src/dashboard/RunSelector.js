import * as React from "react";

export class RunSelector extends React.Component {
    constructor(props) {
        super(props);
        this.state = {value: ''};
        this.handleChange = this.handleChange.bind(this);
    }

    handleChange(event) {
        this.setState({value: event.target.value});
        this.props.loadRunName(event.target.value);
        event.preventDefault();
    }
  
    render() {
        return (
            <label>
                Pick your favorite flavor:
                <select value={this.state.value} onChange={this.handleChange}>
                    <option value="1565514440">1565514440</option>
                    <option value="1565514441">1565514441</option>
                    <option value="1565615243">1565615243</option>
                    <option value="Sint-Maarten-2017v0.4">Sint-Maarten-2017v0.4</option>
                </select>
            </label>
        );
    }
}
