import os
import sys

from model.data import Datasets
from utils import configuration, create_logger, attach_exception_hook
from model.trainer import QuasiSiameseNetwork


def main():
    args = configuration()

    logger = create_logger(__name__)
    sys.excepthook = attach_exception_hook(logger)

    logger.info('START with Configuration:')
    for k, v in sorted(vars(args).items()):
        logger.info('{0}: {1}'.format(k, v))

    qsn = QuasiSiameseNetwork(args)
    datasets = Datasets(args, qsn.transforms)
    save_path = os.path.join(args.checkpointPath, 'best_model_wts.pkl')
    if not args.test:
        qsn.train(args.numberOfEpochs, datasets, args.device, save_path)
    logger.info('Evaluation on test dataset')
    qsn.test(datasets, args.device, save_path)

    logger.info('END')


if __name__ == '__main__':
    main()
