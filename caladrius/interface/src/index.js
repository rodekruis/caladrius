import React from 'react';
import ReactDOM from 'react-dom';
import ScatterPlot from './ScatterPlot.js';
import FullDataTable from './Table.js';
import load_csv_data from './data.js';

class App extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            data: load_csv_data()
        };
    }

    render() {
        return (
            <div className="container">
                <ScatterPlot
                    data={this.state.data}
                    width={300}
                    height={300}
                    />
                <FullDataTable
                  data={this.state.data}
                 />
            </div>
        )
    }
}

// Render application
ReactDOM.render(
    <App />,
    document.getElementById('root')
);
