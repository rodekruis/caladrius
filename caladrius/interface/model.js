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

    get_predictions(model_directory, filename) {
        const prediction_file_path = path.join(
            Config.MODEL_DIRECTORY,
            model_directory,
            "predictions",
            filename
        );
        return fs.promises.readFile(prediction_file_path, "utf8");
    }
}

module.exports = new ModelManager();
