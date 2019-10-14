const fs = require('fs');
const path = require('path');
const DatasetManager = require('./dataset');
const ModelManager = require('./model');

class Server {

    get_dataset_file(req, res) {
        let filename = req.query.filename;
        if(req.query.directory) {
            filename = path.join(req.query.directory, filename);
        }
        DatasetManager
            .get_dataset_file(req.query.name, filename)
                .then(file => res.send(file));
    }

    get_models(req, res) {
        ModelManager
            .get_models()
                .then(files => res.send(files));
    }

    get_predictions(req, res) {
        ModelManager
            .get_predictions(req.query.directory, req.query.filename)
                .then(file => res.send(file));
    }

}

module.exports = new Server();
