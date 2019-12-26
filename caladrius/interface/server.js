const fs = require("fs");
const path = require("path");
const DatasetManager = require("./dataset");
const ModelManager = require("./model");
const Config = require("./config");

class Server {
    get_models(req, res) {
        ModelManager.get_models()
            .then(files => res.send(files))
            .catch(error => {
                res.status(500).send(error);
            });
    }

    get_datasets(req, res) {
        DatasetManager.get_datasets()
            .then(files => res.send(files))
            .catch(error => {
                res.status(500).send(error);
            });
    }

    get_dataset_file(req, res) {
        let filename = Config.DATA_FILENAME;
        if (req.params.filename === Config.DATA_ADMIN_REGIONS_REFERRER) {
            filename = Config.DATA_ADMIN_REGIONS_FILENAME;
        }
        DatasetManager.get_file(req.params.dataset, filename)
            .then(file => res.send(file))
            .catch(error => {
                res.status(500).send(error);
            });
    }

    get_model_predictions(req, res) {
        ModelManager.get_predictions(
            req.params.model,
            req.params.epoch ? parseInt(req.params.epoch) : req.params.epoch
        )
            .then(file => res.send(file))
            .catch(error => {
                res.status(500).send(error);
            });
    }
}

module.exports = new Server();
