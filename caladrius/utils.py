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


def make_directory(directoryPath):
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
    parser.add_argument('--checkpoint-path', type=str, default=os.path.join('.', 'runs'),
                        help='output path')
    parser.add_argument('--data-path', type=str, default=os.path.join('.', 'data', 'Sint-Maarten-2017'),
                        help='data path')
    parser.add_argument('--run-name', type=str, default='{:.0f}'.format(time.time()),
                        help='name to identify execution')
    parser.add_argument('--log-step', type=int, default=100,
                        help='batch step size for logging information')
    parser.add_argument('--number-of-workers', type=int, default=8,
                        help='number of threads used by data loader')

    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable the use of CUDA')
    parser.add_argument('--cuda-device', type=int, default=0,
                        help='specify which GPU to use')
    parser.add_argument('--torch-seed', type=int,
                        help='set a torch seed', default=42)

    parser.add_argument('--input-size', type=int, default=32,
                        help='extent of input layer in the network')
    parser.add_argument('--number-of-epochs', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate for training')

    parser.add_argument('--test', action='store_true', default=False,
                        help='test the model on the test set instead of training')
    parser.add_argument('--max-data-points', default=None, type=int,
                        help='limit the total number of data points used, for debugging on GPU-less laptops')
    parser.add_argument('--accuracy-threshold', type=float, default=0.1,
                        help='window size to calculate regression accuracy')

    args = parser.parse_args()

    arg_vars = vars(args)

    if args.torch_seed is not None:
        torch.manual_seed(arg_vars['torch_seed'])
    else:
        arg_vars['torch_seed'] = torch.initial_seed()

    if args.max_data_points is not None:
        arg_vars['run_name'] = '{}-max_data_points_{}'.format(arg_vars['run_name'], arg_vars['max_data_points'])

    checkpointFolderName = '{}-input_size_{}-learning_rate_{}-batch_size_{}'.format(
        arg_vars['run_name'],
        arg_vars['input_size'],
        arg_vars['learning_rate'],
        arg_vars['batch_size']
    )

    arg_vars['checkpoint_path'] = make_directory(os.path.join(
        arg_vars['checkpoint_path'], checkpointFolderName))
    arg_vars['prediction_path'] = make_directory(os.path.join(
        arg_vars['checkpoint_path'], 'predictions'))
    arg_vars['model_path'] = os.path.join(arg_vars['checkpoint_path'], 'best_model_wts.pkl')

    if torch.cuda.is_available() and not arg_vars['disable_cuda']:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        arg_vars['device'] = torch.device(
            'cuda:{}'.format(arg_vars['cuda_device']))
    else:
        arg_vars['device'] = torch.device('cpu')

    return args


def attach_exception_hook(logger):
    def exception_logger(exceptionType, exceptionValue, exceptionTraceback):
        logger.error('Uncaught Exception', exc_info=(exceptionType, exceptionValue, exceptionTraceback))
    return exception_logger


def create_logger(module_name):
    args = configuration()

    debug_filehandler = logging.FileHandler(os.path.join(args.checkpoint_path, 'run_debug.log'))
    info_filehandler = logging.FileHandler(os.path.join(args.checkpoint_path, 'run_info.log'))

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

