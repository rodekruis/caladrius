import os
import sys
import time
import argparse
import pickle
import logging
import re
import json

import torch

NEURAL_MODELS = ["inception", "light", "after", "shared", "vgg"]
STATISTICAL_MODELS = ["average", "random"]

# logging

logging.getLogger("Fiona").setLevel(logging.ERROR)
logging.getLogger("fiona.collection").setLevel(logging.ERROR)
logging.getLogger("rasterio").setLevel(logging.ERROR)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)


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
            if hasattr(value, "keys"):
                value = dotdict(value)
            self[key] = value


def make_directory(directoryPath):
    if not os.path.isdir(directoryPath):
        os.makedirs(directoryPath)
    return directoryPath


def save_obj(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def readable_float(number):
    return round(float(number), 4)


def dynamic_report_key(label, prefix, condition):
    return "{}{}".format((prefix + "_model_") if condition else "", label)


def run_name_type(run_name):
    run_name = str(run_name)
    pattern = re.compile(r"^[a-zA-Z0-9_\.]{3,30}$")
    if not pattern.match(run_name):
        raise argparse.ArgumentTypeError(
            "Run name can contain only "
            + "alphanumeric, underscore (_) and dot (.) characters. "
            + "Must be at least 3 characters and at most 30 characters long."
        )
    return run_name


def configuration():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General arguments
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=os.path.join(".", "runs"),
        help="output path",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=os.path.join(".", "data", "Sint-Maarten-2017"),
        help="data path",
    )
    parser.add_argument(
        "--label-file", type=str, default="labels.txt", help="filename of labels",
    )
    parser.add_argument(
        "--run-name",
        type=run_name_type,
        default="{:.0f}".format(time.time()),
        help="name to identify execution",
    )
    parser.add_argument(
        "--log-step",
        type=int,
        default=10,
        help="batch step size for logging information",
    )
    parser.add_argument(
        "--number-of-workers",
        type=int,
        default=8,
        help="number of threads used by data loader",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=NEURAL_MODELS[0],
        choices=NEURAL_MODELS + STATISTICAL_MODELS,
        help="type of model",
    )

    parser.add_argument(
        "--disable-cuda", action="store_true", help="disable the use of CUDA"
    )
    parser.add_argument(
        "--cuda-device", type=int, default=0, help="specify which GPU to use"
    )
    parser.add_argument("--torch-seed", type=int, help="set a torch seed", default=42)

    parser.add_argument(
        "--input-size",
        type=int,
        default=32,
        help="extent of input layer in the network",
    )
    parser.add_argument(
        "--number-of-epochs",
        type=int,
        default=100,
        help="number of epochs for training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="learning rate for training"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="test the model on the test set instead of training",
    )
    parser.add_argument(
        "--inference",
        action="store_true",
        default=False,
        help="test and use the model on the inference set instead of training",
    )
    parser.add_argument(
        "--max-data-points",
        default=None,
        type=int,
        help="limit the total number of data points used, for debugging on GPU-less laptops",
    )
    parser.add_argument(
        "--output-type",
        type=str,
        default="classification",
        choices=["regression", "classification"],
        help="choose if want regression or classification model",
    )

    parser.add_argument(
        "--number-classes", type=int, default=4,
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        default="f1_macro",
        choices=[
            "recall_micro",
            "recall_macro",
            "precision_macro",
            "f1_macro",
            "recall_weighted",
            "precision_weighted",
            "f1_weighted",
        ],
        help="choose metric to use for tracking best model",
    )

    parser.add_argument(
        "--test-epoch",
        default=False,
        action="store_true",
        help="If true, run model on test set every epoch. For research purposes.",
    )

    parser.add_argument(
        "--sample-data",
        default=False,
        action="store_true",
        help="If true, resample data such that classes are balanced. For research purposes.",
    )

    parser.add_argument(
        "--weighted-loss",
        type=str,
        default=None,
        choices=["numsamples", "effnumsamples", "medianperc"],
        help="choose weighting strategy for loss function (default: none)",
    )

    parser.add_argument(
        "--freeze",
        default=False,
        action="store_true",
        help="If true, Inception part will not be retrained.",
    )

    parser.add_argument(
        "--no-augment",
        default=False,
        action="store_true",
        help="If False, no augmentations will be applied to the data.",
    )

    parser.add_argument(
        "--augment-type",
        type=str,
        default="original",
        choices=["original", "paper", "equalization"],
        help="choose which data augmentation steps should be applied",
    )

    parser.add_argument(
        "--save-all",
        default=False,
        action="store_true",
        help="If True, whole model will be saved not only state dict. Only for testing purposes",
    )

    parser.add_argument(
        "--probability",
        default=False,
        action="store_true",
        help="If True, probabilistic predictions will be given",
    )

    args = parser.parse_args()

    arg_vars = vars(args)
    arg_vars["model_name"] = arg_vars["run_name"]

    if args.torch_seed is not None:
        torch.manual_seed(arg_vars["torch_seed"])
    else:
        arg_vars["torch_seed"] = torch.initial_seed()

    if args.max_data_points is not None:
        arg_vars["run_name"] = "{}_max_data_points_{}".format(
            arg_vars["run_name"], arg_vars["max_data_points"]
        )

    arg_vars[
        "model_directory"
    ] = "{}-input_size_{}-learning_rate_{}-batch_size_{}".format(
        arg_vars["run_name"],
        arg_vars["input_size"],
        arg_vars["learning_rate"],
        arg_vars["batch_size"],
    )

    arg_vars["checkpoint_path"] = make_directory(
        os.path.join(arg_vars["checkpoint_path"], arg_vars["model_directory"])
    )
    arg_vars["prediction_path"] = make_directory(
        os.path.join(arg_vars["checkpoint_path"], "predictions")
    )
    arg_vars["model_path"] = os.path.join(
        arg_vars["checkpoint_path"], "best_model_wts.pkl"
    )
    arg_vars["run_report_path"] = os.path.join(
        arg_vars["checkpoint_path"], "run_report.json"
    )
    arg_vars["statistical_model"] = arg_vars["model_type"] in STATISTICAL_MODELS
    arg_vars["neural_model"] = arg_vars["model_type"] in NEURAL_MODELS

    if torch.cuda.is_available() and not arg_vars["disable_cuda"]:
        arg_vars["device"] = torch.device("cuda:{}".format(arg_vars["cuda_device"]))
    else:
        arg_vars["device"] = torch.device("cpu")

    return args


def load_run_report(run_report_path):
    run_report_json = dotdict({})
    if os.path.exists(run_report_path):
        with open(run_report_path, "r") as run_report_file:
            run_report_json = json.load(run_report_file)
            run_report_json = dotdict(run_report_json)
            run_report_json.device = torch.device(run_report_json.device)
    return run_report_json


def save_run_report(run_report_json):
    run_report_json.device = str(run_report_json.device)
    with open(run_report_json.run_report_path, "w") as run_report_file:
        json.dump(run_report_json, run_report_file, indent=4)


def attach_exception_hook(logger):
    def exception_logger(exceptionType, exceptionValue, exceptionTraceback):
        logger.error(
            "Uncaught Exception",
            exc_info=(exceptionType, exceptionValue, exceptionTraceback),
        )

    return exception_logger


def create_logger(module_name):
    args = configuration()

    debug_filehandler = logging.FileHandler(
        os.path.join(args.checkpoint_path, "run_debug.log")
    )
    info_filehandler = logging.FileHandler(
        os.path.join(args.checkpoint_path, "run_info.log")
    )

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
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
