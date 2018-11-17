import os
import sys

# from model import Model
# from network import Siamese
from data import loadDataset
from utils import configuration

import logging

# logging

logger = logging.getLogger(__name__)
logging.getLogger('Fiona').setLevel(logging.ERROR)
logging.getLogger('fiona.collection').setLevel(logging.ERROR)
logging.getLogger('rasterio').setLevel(logging.ERROR)


def exceptionLogger(exceptionType, exceptionValue, exceptionTraceback):
    logger.error("Uncaught Exception", exc_info=(exceptionType, exceptionValue, exceptionTraceback))


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

    # Load Dataset
    dataset = loadDataset(args, transforms=None)

    # Initialize Model
    # model = Model(Siamese, data, args)

    # if args.train:
    #     trainResult = model.train(save=True)
    #     # plt.plot(trainResult)
    #     # plt.savefig(os.path.join(args.checkpointPath, 'trainingLoss.png'))

    # model.eval()
    # if args.eval is not None:
    #     model.predict()

    logger.info('END')
