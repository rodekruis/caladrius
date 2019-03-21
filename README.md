# Caladrius - Assessing Building Damage caused by Natural Disasters using Satellite Images
## Created by: Artificial Incompetence for the Red Cross #1 Challenge in the 2018 Hackathon for Peace, Justice and Security

## Network Architecture

The network architecture is a pseudo-siamese network with two ImageNet pre-trained Inception_v3 models.


## Setup

#### Requirements:
- Python 3.6.5
- Install the required libraries:

```
pip install -r requirements.txt
yarn install
```

## Dataset

The dataset can be downloaded from [here](http://gulfaraz.com/share/rc.tgz "RC Challenge 1 Raw Dataset").

Extract the contents to the `data` folder.

To create the dataset execute `python sint-maarten-2017.py`.

This will create the dataset as per the [specifications](DATASET.md).

## Execute

##### Training:

```
python run.py --runName caladrius_2019
```

##### Testing:

```
python run.py --runName caladrius_2019 --test
```


## Configuration
There are several parameters, that can be set, the full list is the following:

```
usage: run.py [-h] [--checkpointPath CHECKPOINTPATH] [--dataPath DATAPATH]
              [--runName RUNNAME] [--logStep LOGSTEP]
              [--numberOfWorkers NUMBEROFWORKERS] [--disableCuda]
              [--cudaDevice CUDADEVICE] [--torchSeed TORCHSEED]
              [--inputSize INPUTSIZE] [--numberOfEpochs NUMBEROFEPOCHS]
              [--batchSize BATCHSIZE] [--learningRate LEARNINGRATE] [--test]

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
