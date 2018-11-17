import os
import argparse

import torch


class dotdict(dict):
    """
    a dictionary that supports dot notation
    as well as dictionary access notation
    usage: d = dotdict() or d = dotdict({'val1':'first'})
    set attributes: d.val2 = 'second' or d['val2'] = 'second'
    get attributes: d.val2 or d['val2']
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        super().__init__()
        for key, value in dct.items():
            if hasattr(value, 'keys'):
                value = dotdict(value)
            self[key] = value


def makeDirectory(directoryPath):
    if not os.path.isdir(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath


def configuration():
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument('--checkpointPath', type=str, default=os.path.join('.', 'temp'),
                        help='output path')
    parser.add_argument('--dataPath', type=str, default=os.path.join('.', 'data', 'RC Challenge 1', '1'),
                        help='data path')
    parser.add_argument('--datasetName', type=str, default='train',
                        choices=['train', 'test_1', 'test_2'],
                        help='name of dataset to use')
    parser.add_argument('--logStep', type=int, default=100,
                        help='batch step size for logging information')
    parser.add_argument('--numberOfWorkers', type=int, default=4,
                        help='number of threads used by data loader')

    parser.add_argument('--disableCuda', action='store_true',
                        help='disable the use of CUDA')
    parser.add_argument('--cudaDevice', type=int, default=0,
                        help='specify which GPU to use')
    parser.add_argument('--torchSeed', type=int,
                        help='set a torch seed', default=42)

    parser.add_argument('--inputSize', type=int, default=32,
                        help='extent of input layer in the network')
    parser.add_argument('--numberOfEpochs', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--batchSize', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--learningRate', type=float, default=0.001,
                        help='learning rate for training')

    parser.add_argument("--outputType", type=str,
                        choices={"soft-targets", "softmax"}, help="influences the output of the model",
                        default="softmax")

    parser.add_argument("--networkType", type=str,
                        choices={"pre-trained", "full"}, help="type of network to train",
                        default="pre-trained")

    args = parser.parse_args()

    arg_vars = vars(args)

    if args.torchSeed is not None:
        torch.manual_seed(arg_vars['torchSeed'])
    else:
        arg_vars['torchSeed'] = torch.initial_seed()

    checkpointFolderName = '{}-{}-{}'.format(
        arg_vars['datasetName'],
        arg_vars['inputSize'],
        arg_vars['learningRate']
    )

    arg_vars['checkpointPath'] = makeDirectory(os.path.join(
        arg_vars['checkpointPath'], checkpointFolderName))

    if torch.cuda.is_available() and not arg_vars['disableCuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        arg_vars['device'] = torch.device(
            'cuda:{}'.format(arg_vars['cudaDevice']))
    else:
        arg_vars['device'] = torch.device('cpu')

    return args
