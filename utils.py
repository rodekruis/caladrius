import os
import sys
import time
import argparse
import pickle
import logging

import torch

# logging

logging.getLogger('Fiona').setLevel(logging.ERROR)
logging.getLogger('fiona.collection').setLevel(logging.ERROR)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)


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


def save_obj(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def configuration():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General arguments
    parser.add_argument('--checkpointPath', type=str, default=os.path.join('.', 'runs'),
                        help='output path')
    parser.add_argument('--dataPath', type=str, default=os.path.join('.', 'data', 'Sint-Maarten-2017'),
                        help='data path')
    parser.add_argument('--runName', type=str, default='{:.0f}'.format(time.time()),
                        help='name to identify execution')
    parser.add_argument('--logStep', type=int, default=100,
                        help='batch step size for logging information')
    parser.add_argument('--numberOfWorkers', type=int, default=8,
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

    parser.add_argument('--test', action='store_true', default=False,
                        help='test the model on the test set instead of training')

    args = parser.parse_args()

    arg_vars = vars(args)

    if args.torchSeed is not None:
        torch.manual_seed(arg_vars['torchSeed'])
    else:
        arg_vars['torchSeed'] = torch.initial_seed()

    checkpointFolderName = '{}-input_size_{}-learning_rate_{}-batch_size_{}'.format(
        arg_vars['runName'],
        arg_vars['inputSize'],
        arg_vars['learningRate'],
        arg_vars['batchSize']
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


def attach_exception_hook(logger):
    def exception_logger(exceptionType, exceptionValue, exceptionTraceback):
        logger.error('Uncaught Exception', exc_info=(exceptionType, exceptionValue, exceptionTraceback))
    return exception_logger


def create_logger(module_name):
    args = configuration()

    debug_filehandler = logging.FileHandler(os.path.join(args.checkpointPath, 'run_debug.log'))
    info_filehandler = logging.FileHandler(os.path.join(args.checkpointPath, 'run_info.log'))

    formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    debug_filehandler.setFormatter(formatter)
    info_filehandler.setFormatter(formatter)

    debug_filehandler.setLevel(logging.DEBUG)
    info_filehandler.setLevel(logging.INFO)

    streamhandler = logging.StreamHandler(sys.stdout)
    streamhandler.setFormatter(formatter)
    streamhandler.setLevel(logging.DEBUG)

    logger = logging.getLogger(module_name)

    logger.addHandler(debug_filehandler)
    logger.addHandler(info_filehandler)
    logger.addHandler(streamhandler)

    logger.setLevel(logging.DEBUG)

    return logger

