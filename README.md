[![version: 0.6.2](https://img.shields.io/badge/version-0.6.2-ED2E26.svg?style=flat-square)](https://github.com/rodekruis/caladrius)
[![code style: prettier](https://img.shields.io/badge/code_style-prettier-ff69b4.svg?style=flat-square)](https://github.com/prettier/prettier)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

# [Caladrius](https://en.wikipedia.org/wiki/Caladrius) - Assessing Building Damage caused by Natural Disasters using Satellite Images
## Created by: Artificial Incompetence for the Red Cross #1 Challenge in the 2018 Hackathon for Peace, Justice and Security

## Network Architecture

The network architecture is a pseudo-siamese network with two ImageNet pre-trained Inception_v3 models.

## Using Docker

Install [Docker](https://www.docker.com/get-started).

Download the [Caladrius Docker Image](https://hub.docker.com/r/gulfaraz/caladrius) using,

```
docker pull gulfaraz/caladrius
```

Create a [data](#dataset) folder in your local machine.

Create a docker container using,

```
docker run --name caladrius -dit -v <path/to/data>:/workspace/data -p 5000:5000 gulfaraz/caladrius
```

Access the container using,

```
docker exec -it caladrius bash
```

## Manual Setup

#### Requirements:

- [Python 3.6.5 or higher](https://www.python.org/downloads/)
- [Anaconda or Miniconda 2019.07 or higher](https://www.anaconda.com/distribution/#download-section)
- [NodeJS v10 or higher](https://nodejs.org/en/download/)
- Run the following script,

```bash
./caladrius_install.sh
```

## Dataset - Sint Maarten 2017 Hurricane Irma

##### 1. Download Raw Dataset:
The Sint Maarten 2017 dataset can be downloaded from [here](https://rodekruis.sharepoint.com/sites/510-Team/Gedeelde%20%20documenten/%5BPRJ%5D%20Automated%20Damage%20Assessment/DATASET/Sint-Maarten-2017/rc.tgz "RC Challenge 1 Raw Dataset").

##### 2. Extract Raw Dataset:

To extract the contents to the `data` folder execute,

```
tar -xvzf rc.tgz
```

##### 3. Create Training Dataset:

Transform the raw dataset to a training dataset using,

```
python caladrius/sint_maarten_2017.py --run-all
```

The above command will create the dataset as per the [specifications](DATASET.md).

##### Configuration:

`sint_maarten_2017.py` accepts the command line arguments described below,

```
usage: sint_maarten_2017.py [-h] [--run-all] [--create-image-stamps]
                      [--query-address-api] [--address-api ADDRESS_API]
                      [--address-api-key ADDRESS_API_KEY]
                      [--create-report-info-file]

optional arguments:
  -h, --help            show this help message and exit
  --run-all             Run all of the steps: create and split image stamps,
                        query for addresses, and create information file for
                        the report. Overrides individual step flags. (default:
                        False)
  --create-image-stamps
                        For each building shape, creates a before and after
                        image stamp for the learning model, and places them in
                        the approriate directory (train, validation, or test)
                        (default: False)
  --query-address-api   For each building centroid, preforms a reverse geocode
                        query and stores the address in a cache file (default:
                        False)
  --address-api ADDRESS_API
                        Which API to use for the address query (default:
                        openmapquest)
  --address-api-key ADDRESS_API_KEY
                        Some APIs (like OpenMapQuest) require an API key
                        (default: None)
  --create-report-info-file
                        Creates a geojson file that contains the locations and
                        shapes of the buildings, their respective
                        administrative regions and addresses (if --query-
                        address-api has been run) (default: False)
```

## Interface

From the `caladrius/interface` directory execute,

```
npm start
```

The interface should be accessible at `http://localhost:5000`.

## Model

##### Training:

```
python caladrius/run.py --run-name caladrius_2019
```

##### Testing:

```
python caladrius/run.py --run-name caladrius_2019 --test
```

[Click here to download the trained model.](https://drive.google.com/open?id=1zdWhefcjWto8CxWAR75xO_yMBEiq1QLx)


## Configuration

`run.py` accepts the command line arguments described below,

```
usage: run.py [-h] [--checkpoint-path CHECKPOINT_PATH] [--data-path DATA_PATH]
              [--run-name RUN_NAME] [--log-step LOG_STEP]
              [--number-of-workers NUMBER_OF_WORKERS] [--disable-cuda]
              [--cuda-device CUDA_DEVICE] [--torch-seed TORCH_SEED]
              [--input-size INPUT_SIZE] [--number-of-epochs NUMBER_OF_EPOCHS]
              [--batch-size BATCH_SIZE] [--learning-rate LEARNING_RATE]
              [--test] [--max-data-points MAX_DATA_POINTS]
              [--accuracy-threshold ACCURACY_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  --checkpoint-path CHECKPOINT_PATH
                        output path (default: ./runs)
  --data-path DATA_PATH
                        data path (default: ./data/Sint-Maarten-2017)
  --run-name RUN_NAME   name to identify execution (default: <timestamp>)
  --log-step LOG_STEP   batch step size for logging information (default: 100)
  --number-of-workers NUMBER_OF_WORKERS
                        number of threads used by data loader (default: 8)
  --disable-cuda        disable the use of CUDA (default: False)
  --cuda-device CUDA_DEVICE
                        specify which GPU to use (default: 0)
  --torch-seed TORCH_SEED
                        set a torch seed (default: 42)
  --input-size INPUT_SIZE
                        extent of input layer in the network (default: 32)
  --number-of-epochs NUMBER_OF_EPOCHS
                        number of epochs for training (default: 100)
  --batch-size BATCH_SIZE
                        batch size for training (default: 32)
  --learning-rate LEARNING_RATE
                        learning rate for training (default: 0.001)
  --test                test the model on the test set instead of training
                        (default: False)
  --max-data-points MAX_DATA_POINTS
                        limit the total number of data points used, for
                        debugging on GPU-less laptops (default: None)
  --accuracy-threshold ACCURACY_THRESHOLD
                        window size to calculate regression accuracy (default:
                        0.1)
```

## Development

When making changes, increment `VERSION` according to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and update `CHANGES.md`.

## Docker image

To build and tag the Docker image with `VERSION`,

```bash
make build_production
```

For development and tagging with the latest commit version,

```bash
make build_fast
```
