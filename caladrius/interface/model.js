const fs = require("fs");
const path = require("path");
const Config = require("./config");

class ModelManager {
    read_model_report(model_report_path) {
        return new Promise((resolve, reject) => {
            fs.exists(model_report_path, exists => {
                if (exists) {
                    fs.readFile(
                        model_report_path,
                        (error, model_report_raw) => {
                            if (error) resolve(null);
                            const model_report_json = JSON.parse(
                                model_report_raw
                            );
                            resolve(model_report_json);
                        }
                    );
                } else {
                    resolve(null);
                }
            });
        });
    }

    get_models() {
        return new Promise((resolve, reject) => {
            fs.readdir(Config.MODEL_DIRECTORY, (error, models) => {
                if (error) {
                    reject(error);
                } else {
                    const promises = [];
                    models.forEach(model_name => {
                        const model_report_path = path.join(
                            Config.MODEL_DIRECTORY,
                            model_name,
                            Config.MODEL_REPORT_FILENAME
                        );
                        promises.push(
                            this.read_model_report(model_report_path)
                        );
                    });
                    Promise.all(promises).then(models => {
                        resolve(
                            models
                                .filter(model => model && model.test)
                                .sort(
                                    (model_a, model_b) =>
                                        new Date(model_b.test_end_time) -
                                        new Date(model_a.test_end_time)
                                )
                        );
                    });
                }
            });
        });
    }

    read_predictions(prediction_file_path, split, epoch) {
        return new Promise((resolve, reject) => {
            if (split === "train") {
                resolve(null);
            } else {
                fs.exists(prediction_file_path, exists => {
                    if (exists) {
                        fs.readFile(
                            prediction_file_path,
                            "utf8",
                            (error, prediction_file_raw) => {
                                if (error) resolve(null);
                                resolve({
                                    split: split,
                                    epoch: epoch,
                                    predictions: prediction_file_raw,
                                });
                            }
                        );
                    } else {
                        resolve(null);
                    }
                });
            }
        });
    }

    get_predictions(model_name) {
        const predictions_directory = path.join(
            Config.MODEL_DIRECTORY,
            model_name,
            Config.PREDICTIONS_DIRECTORY
        );
        return new Promise((resolve, reject) => {
            fs.readdir(predictions_directory, (error, prediction_filenames) => {
                if (error) {
                    reject(error);
                } else {
                    const promises = [];
                    prediction_filenames.forEach(prediction_filename => {
                        const prediction_file_path = path.join(
                            predictions_directory,
                            prediction_filename
                        );
                        const prediction_filename_parts = prediction_filename.split(
                            "-"
                        );
                        const split = prediction_filename_parts[1].split(
                            "_"
                        )[1];
                        const epoch = parseInt(
                            prediction_filename_parts[2].split("_")[1]
                        );
                        promises.push(
                            this.read_predictions(
                                prediction_file_path,
                                split,
                                epoch
                            )
                        );
                    });
                    Promise.all(promises).then(predictions => {
                        predictions = predictions.filter(
                            prediction => prediction
                        );
                        let prediction_object = {
                            validation: [],
                            test: [],
                            inference: [],
                        };
                        predictions.forEach(prediction => {
                            prediction_object[prediction["split"]][
                                prediction["epoch"] - 1
                            ] = prediction["predictions"];
                        });
                        resolve(prediction_object);
                    });
                }
            });
        });
    }
}

module.exports = new ModelManager();
