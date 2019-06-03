# [Caladrius](https://en.wikipedia.org/wiki/Caladrius) - Assessing Building Damage caused by Natural Disasters using Satellite Images
## Created by: Artificial Incompetence for the Red Cross #1 Challenge in the 2018 Hackathon for Peace, Justice and Security

## Network Architecture

The network architecture is a pseudo-siamese network with two ImageNet pre-trained Inception_v3 models.

## Using Docker

Install [Docker](https://www.docker.com/get-started).

Download the [Caladrius Docker Image](https://hub.docker.com/r/gulfaraz/caladrius) using `docker pull gulfaraz/caladrius`.

Create a [data](#dataset) folder in your local machine.

Create a docker container using `docker run --name caladrius -dit -v <path/to/data>:/workspace/data gulfaraz/caladrius`.

Access the container using `docker exec -it caladrius bash`.


## Manual Setup

#### Requirements:
- Python 3.6.5
- [Yarn](https://yarnpkg.com/)
- Install the required libraries:

```
pip install -r requirements.txt
pip install .
yarn install
```

## Dataset

The dataset can be downloaded from [here](http://gulfaraz.com/share/rc.tgz "RC Challenge 1 Raw Dataset").

Extract the contents to the `data` folder. (Default Path: `./data`)

To create the initial dataset, execute `caladrius_data --run-all`.
This will create the dataset as per the [specifications](DATASET.md).

There are several parameters that you can specify, described below:
```
usage: caladrius_data [-h] [--run-all] [--create-image-stamps]
                      [--query-address-api] [--address-api ADDRESS_API]
                      [--address-api-key ADDRESS_API_KEY]
                      [--create-report-info-file]

optional arguments:
  -h, --help            show this help message and exit
  --run-all             Run all of the steps: create and split image stamps,
                        query for addresses, and create information file for
                        thereport. Overrides individual step flags. (default:
                        False)
  --create-image-stamps
                        For each building shape, creates a before and
                        after image stamp for the learning model, and places
                        them in the approriate directory (train, validation,
                        or test) (default: False)
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
                        administrativeregions and addresses (if --query-
                        address-api has been run) (default: False)
```



## Interface

Execute `python -m http.server` in the root directory to create a local server.

The interface can be accessed using `http://localhost:8000/caladrius/interface/`.

## Model

##### Training:

```
caladrius --runName caladrius_2019
```

##### Testing:

```
caladrius --runName caladrius_2019 --test
```

[Click here to download the trained model.](https://drive.google.com/open?id=1jMhEpA_czv2d-84Ym4Nm8LP0TJnhzvnq)


## Configuration
There are several parameters, that can be set, the full list is the following:

```
usage: caladrius [-h] [--checkpointPath CHECKPOINTPATH] [--dataPath DATAPATH]
                 [--runName RUNNAME] [--logStep LOGSTEP]
                 [--numberOfWorkers NUMBEROFWORKERS] [--disableCuda]
                 [--cudaDevice CUDADEVICE] [--torchSeed TORCHSEED]
                 [--inputSize INPUTSIZE] [--numberOfEpochs NUMBEROFEPOCHS]
                 [--batchSize BATCHSIZE] [--learningRate LEARNINGRATE]
                 [--test] [--maxDataPoints MAXDATAPOINTS]
                 
optional arguments:
  -h, --help            show this help message and exit
  --checkpointPath CHECKPOINTPATH
                        output path (default: ./runs)
  --dataPath DATAPATH   data path (default: ./data/Sint-Maarten-2017)
  --runName RUNNAME     name to identify execution (default: <timestamp>)
  --logStep LOGSTEP     batch step size for logging information (default: 100)
  --numberOfWorkers NUMBEROFWORKERS
                        number of threads used by data loader (default: 8)
  --disableCuda         disable the use of CUDA (default: False)
  --cudaDevice CUDADEVICE
                        specify which GPU to use (default: 0)
  --torchSeed TORCHSEED
                        set a torch seed (default: 42)
  --inputSize INPUTSIZE
                        extent of input layer in the network (default: 32)
  --numberOfEpochs NUMBEROFEPOCHS
                        number of epochs for training (default: 100)
  --batchSize BATCHSIZE
                        batch size for training (default: 32)
  --learningRate LEARNINGRATE
                        learning rate for training (default: 0.001)
  --test                test the model on the test set instead of training
                        (default: False)
```

## Development

When making changes, increment `VERSION` according to [PEP 440](https://www.python.org/dev/peps/pep-0440/)
and update `CHANGES.md`


## Docker image

To build and tag the Docker image with `VERSION`, use:
```bash
make build_production
```
For development and tagging with the latest commit version:
```bash
make build_fast
```
