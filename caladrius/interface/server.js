const fs = require("fs");
const path = require("path");
const DatasetManager = require("./dataset");
const ModelManager = require("./model");

class Server {
    get_dataset_file(req, res) {
        let filename = req.query.filename;
        if (req.query.directory) {
            filename = path.join(req.query.directory, filename);
        }
        DatasetManager.get_dataset_file(req.query.name, filename)
            .then(file => res.send(file))
            .catch(error => {
                res.status(500).send(error);
            });
    }

    get_models(req, res) {
        ModelManager.get_models()
            .then(files => res.send(files))
            .catch(error => {
                res.status(500).send(error);
            });
    }

    get_predictions(req, res) {
        ModelManager.get_predictions(req.query.directory, req.query.filename)
            .then(file => res.send(file))
            .catch(error => {
                res.status(500).send(error);
            });
    }
}

module.exports = new Server();
