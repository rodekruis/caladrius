const fs = require("fs");
const path = require("path");

// these need to be manually configured
const DATASET_DIRECTORY = "../../data";

class DatasetManager {
    get_dataset_file(dataset_name, filename) {
        const dataset_directory = path.join(DATASET_DIRECTORY, dataset_name);
        const file_path = path.join(dataset_directory, filename);
        return fs.promises.readFile(file_path, "utf8");
    }
}

module.exports = new DatasetManager();
