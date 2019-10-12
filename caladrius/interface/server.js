const fs = require('fs');
const path = require('path');
const DatasetManager = require('./dataset');

class Server {
    get_dataset_file(req, res) {
        let filename = req.query.filename;
        if(req.query.directory) {
            filename = path.join(req.query.directory, filename);
        }
        DatasetManager
            .get_dataset_file(req.query.name, filename)
                .then(
                    (file) => {
                        res.send(file);
                    }
                );
    }
}

module.exports = new Server();
