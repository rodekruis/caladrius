import React from 'react';
import ReactDOM from 'react-dom';
import ScatterPlot from './ScatterPlot.js';
import FullDataTable from './Table.js';

ReactDOM.render(
  <div>
    <ScatterPlot width={800} height={600} />
    <FullDataTable/>
  </div>,
  document.getElementById('root')
);
