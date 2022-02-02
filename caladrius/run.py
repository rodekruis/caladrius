import os
import sys

from model.data import Datasets
from utils import (
    configuration,
    create_logger,
    attach_exception_hook,
    load_run_report,
    save_run_report,
    dotdict,
)
from model.trainer import QuasiSiameseNetwork


def main():
    args = configuration()
    run_report = load_run_report(args.run_report_path)

    logger = create_logger(__name__)
    sys.excepthook = attach_exception_hook(logger)

    logger.info("python {}".format(" ".join(sys.argv)))
    logger.info("START with Configuration:")
    for k, v in sorted(vars(args).items()):
        logger.info("{0}: {1}".format(k, v))
        if (k == "model_type") and (args.test or args.inference):
            continue
        run_report[k] = v

    qsn = QuasiSiameseNetwork(args)
    datasets = Datasets(args, qsn.transforms)
    if args.neural_model and not (args.test or args.inference):
        run_report = qsn.train(
            run_report, datasets, args.number_of_epochs, args.selection_metric
        )
    if args.test:
        logger.info("Evaluating on test dataset")
        run_report = qsn.test(run_report, datasets, args.selection_metric)
    if args.inference:
        logger.info("Inference started")
        qsn.inference(datasets)

    save_run_report(run_report)
    logger.info("END")


if __name__ == "__main__":
    main()
