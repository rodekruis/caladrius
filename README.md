# Caladrius - Assessing Building Damage caused by Natural Disasters using Satellite Images
## Created by: Artificial Incompetence for the Red Cross #1 Challenge in the 2018 Hackathon for Peace, Justice and Security

## Approach

We are using (quasi-) Siamese-networks (in the pretrained case the weights are not shared between the twins). There are two seperate Siamese-network instances in this project:

1. A transfer learned model using Inception_v3 (pretrained on ImageNet)
2. A fully trained Siamese network, that is trained only on the challenge's images from scratch



## Setup

#### Requirements:
- Python 3.6.5
- Install the required libraries:

```
pip install -r requirements.txt
```

## How to run
#### Important: the checkpoint paths should be different for the different model types!

### Examples:

#### 1. Inception pretrained network transfer learning
##### *Training* (transfer learning):

```
python run.py --networkType pre-trained --checkpointPath fully-trained-checkpoint
```

##### *Testing* (use the same command with the addition of --test):

```
python run.py --networkType pre-trained --checkpointPath pretrained-checkpoint --test
```

#### 2. Fully trained network
##### *Training*:
```
python run.py --networkType full --checkpointPath fully-trained-checkpoint
```

##### *Testing* (use the same command with the addition of --test):
```
python run.py --networkType full --checkpointPath fully-trained-checkpoint --test
```


## Detailed how to run
There are several parameters, that can be set, the full list is the following:

```
usage: run.py [-h] [--checkpointPath CHECKPOINTPATH] [--dataPath DATAPATH]
              [--datasetName {train,test_1,test_2}] [--logStep LOGSTEP]
              [--numberOfWorkers NUMBEROFWORKERS] [--disableCuda]
              [--cudaDevice CUDADEVICE] [--torchSeed TORCHSEED]
              [--inputSize INPUTSIZE] [--numberOfEpochs NUMBEROFEPOCHS]
              [--batchSize BATCHSIZE] [--learningRate LEARNINGRATE]
              [--outputType {soft-targets,softmax}]
              [--networkType {pre-trained,full}] [--numFreeze NUMFREEZE]
              [--test]

optional arguments:
-h, --help            show this help message and exit
  --checkpointPath CHECKPOINTPATH
                        output path
  --dataPath DATAPATH   data path
  --datasetName {train,test_1,test_2}
                        name of dataset to use
  --logStep LOGSTEP     batch step size for logging information
  --numberOfWorkers NUMBEROFWORKERS
                        number of threads used by data loader
  --disableCuda         disable the use of CUDA
  --cudaDevice CUDADEVICE
                        specify which GPU to use
  --torchSeed TORCHSEED
                        set a torch seed
  --inputSize INPUTSIZE
                        extent of input layer in the network
  --numberOfEpochs NUMBEROFEPOCHS
                        number of epochs for training
  --batchSize BATCHSIZE
                        batch size for training
  --learningRate LEARNINGRATE
                        learning rate for training
  --outputType {soft-targets,softmax}
                        influences the output of the model
  --networkType {pre-trained,full}
                        type of network to train
  --numFreeze NUMFREEZE
                        (- number of layers to not freeze)
  --test                test the model on the test set instead of training
```
