# Artificial Incompetence - Red Cross #1 - damage assessment of buildings after a disaster

## Approach

We are using (quasi-) Siamese-networks (in the pretrained case the weights are not shared between the twins). There are two seperate Siamese-network instances in this project:

1. A fully trained Siamese network, that is trained only on the challenge's images from scratch
2. A transfer learned model using Inception_v3 (pretrained on ImageNet)


## Setup

#### Requirements:
- Python 3.6.5
- virtualenv (optional, but recommended)
- Install the required libraries:

```
pip install -r requirements.txt
```


### Important: the checkpoint paths should be different for the different model types!

### Training in general
#### There are several parameters, that can be set, for the full list see the Detailed how to run section

### Inception pretrained network transfer learning

#### *Testing* the fully trained network (use the same command with the addition of --test):
```
python run.py --networkType pre-trained --checkpointPath pretrained-checkpoint --test
```

### Fully trained network
#### *Training* the fully trained network:
```
python run.py --networkType full --checkpointPath fully-trained-checkpoint
```

#### *Testing* the fully trained network (use the same command with the addition of --test):
```
python run.py --networkType full --checkpointPath fully-trained-checkpoint --test
```


## Detailed how to run

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


## Implementation Notes
We used

## TODO
