const fs = require("fs");
const path = require("path");
const Config = require("./config");

class DatasetManager {
    validate_caladrius_dataset(dataset_coordinates_path) {
        return new Promise((resolve, reject) => {
            fs.exists(dataset_coordinates_path, exists => {
                if (exists) {
                    fs.readFile(
                        dataset_coordinates_path,
                        (error, dataset_coordinates_raw) => {
                            if (error) resolve(null);
                            const dataset_coordinates_json = JSON.parse(
                                dataset_coordinates_raw
                            );
                            resolve(dataset_coordinates_json);
                        }
                    );
                } else {
                    resolve(null);
                }
            });
        });
    }

    get_datasets() {
        return new Promise((resolve, reject) => {
            fs.readdir(Config.DATA_DIRECTORY, (error, datasets) => {
                if (error) {
                    reject(error);
                } else {
                    const promises = [];
                    datasets.forEach(dataset_name => {
                        const dataset_coordinates_path = path.join(
                            Config.DATA_DIRECTORY,
                            dataset_name,
                            Config.DATA_FILENAME
                        );
                        promises.push(
                            this.validate_caladrius_dataset(
                                dataset_coordinates_path
                            )
                        );
                    });
                    Promise.all(promises).then(datasets => {
                        resolve(datasets.filter(dataset => dataset));
                    });
                }
            });
        });
    }

    get_file(dataset_name, filename) {
        const dataset_directory = path.join(
            Config.DATA_DIRECTORY,
            dataset_name
        );
        const file_path = path.join(dataset_directory, filename);
        return fs.promises.readFile(file_path, "utf8");
    }
}

module.exports = new DatasetManager();
