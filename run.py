import os
import sys
import logging

from data import Datasets
from utils import configuration
from trainer import QuasiSiameseNetwork

# logging

logger = logging.getLogger(__name__)
logging.getLogger('Fiona').setLevel(logging.ERROR)
logging.getLogger('fiona.collection').setLevel(logging.ERROR)
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)


def exceptionLogger(exceptionType, exceptionValue, exceptionTraceback):
    logger.error("Uncaught Exception", exc_info=(
        exceptionType, exceptionValue, exceptionTraceback))


sys.excepthook = exceptionLogger


if __name__ == '__main__':
    args = configuration()

    logging.basicConfig(
        handlers=[
            logging.FileHandler(os.path.join(args.checkpointPath, 'run.log')),
            logging.StreamHandler(sys.stdout)
        ],
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s %(message)s'
    )

    logger.info('START with Configuration : {}'.format(args))

    qsn = QuasiSiameseNetwork(args)
    datasets = Datasets(args, qsn.transforms)
    save_path = os.path.join(args.checkpointPath, "best_model_wts.pkl")
    if args.test:
        logger.info("Testing the model")
        qsn.test(datasets, args.device, save_path)
    else:
        qsn.train(args.numberOfEpochs, datasets, args.device, save_path)

    logger.info('END')
