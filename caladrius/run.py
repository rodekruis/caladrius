import os
import sys
import argparse

import torch.cuda

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

#Added Polle
import torch
from model.tsne_cal import run_tsne, make_animation
import pickle
import numpy as np
import time
from baal.utils.plot_utils import make_animation_from_data
import copy
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.WAAL import caladrius_waal_extractor

# Version used for actual tests on virtual GPU, including AL. The AL part can be turned of and on using settings
# similarly to the regular Caladrius run settings.

def main():
    args = configuration()
    print(args)
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

    start_time = time.time()
    qsn = QuasiSiameseNetwork(args) #Initialize empty network; possibly include pretrained model training here such
    # that the model is not loaded twice, making it somewhat faster

    #Change Polle
    if args.pretrained_model_path: #This loads the pretrained model which is to be finetuned
        if torch.cuda.is_available():  # Note: This assumes the loaded model is trained on GPU and thus in cuda type, if this is not the case loaction must be mapped to cuda
            qsn.model.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, args.pretrained_model_name)))
        else:
            qsn.model.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, args.pretrained_model_name), map_location=torch.device('cpu')))

    if args.reinitialize_model: #todo; NOT RELEVANT
        qsn_model_original = copy.deepcopy(qsn.model)

    if args.active == 'WAAL': #Todo: not relevant #Divide model in three pasrts: net_fea, net_clf and net_dis
        qsn = caladrius_waal_extractor(qsn)

        #below to use layer norm instead of batch norm
        qsn.net_dis.similarity.bn_0 = torch.nn.LayerNorm(qsn.net_dis.similarity.layer_0.out_features)
        qsn.net_dis.similarity.bn_1 = torch.nn.LayerNorm(qsn.net_dis.similarity.layer_1.out_features)



    datasets = Datasets(args, qsn.transforms)
    if args.neural_model and not (args.test or args.inference):
        #Change Polle
        if args.active:
            active_learning_ds = False
            last_query = False
            if args.pretrained_model_path:
                run_report = qsn.validation_pre_trained(run_report, datasets,
                                                    args.selection_metric)  # to first find the validation predictions for the pre-trained model
            for i in range(args.active_iterations+1):
                #Just added for comparing restart part
                if i==0: #Todo: not relevant or adjust
                    if args.active == 'WAAL' and False: #Todo: adjust here for first epoch adjustions
                        num_epochs = 1
                    elif args.number_of_epochs==5: #for full waal type
                        num_epochs = args.number_of_epochs
                    elif args.active_images_per_iteration==100:
                        num_epochs = args.number_of_epochs
                    elif args.active=='WAAL' and args.number_of_epochs==100:
                        num_epochs=50
                    else:
                        num_epochs = 20
                else: #todo: outside of else
                    num_epochs = args.number_of_epochs
                run_report, active_learning_ds = qsn.train(
                    run_report, datasets, num_epochs, args.selection_metric, active_learning_ds,last_query, i
                )
                if i == (args.active_iterations - 1): #added such that after the last query, a new model is still trained but no new query is undertaken
                    last_query = True
                    if args.num_epochs_last_actiter:  # To set number of epochs to last iteration value, if this is parsed
                        num_epochs = args.num_epochs_last_actiter
                if i != args.active_iterations and args.reinitialize_model == True: #Todo: not relevant for now # reinitialize model between active iterations
                    qsn.model = copy.deepcopy(qsn_model_original)
                    qsn.optimizer = Adam(qsn.model.parameters(), lr=qsn.lr)
                    qsn.lr_scheduler = ReduceLROnPlateau(
                        qsn.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
                    )
        else:
            if args.pretrained_model_path:
                run_report = qsn.validation_pre_trained(run_report, datasets,
                                                    args.selection_metric)
            run_report = qsn.train(
                run_report, datasets, args.number_of_epochs, args.selection_metric
            )  # Train the empty dataset
    logger.info("Evaluating on test dataset")
    if args.active:
        #In this case, we did not have sufficient valid. data so simply pick th elast created model. NOTE: this was not
        # yet implemented when doing multiple runs on Random, BALD and BatchBALD
        best_model_wts = copy.deepcopy(qsn.model.state_dict())
        torch.save(best_model_wts, qsn.model_path)
    run_report = qsn.test(run_report, datasets, args.selection_metric) #test the model performance
    if args.inference:
        logger.info("Inference started")
        qsn.inference(datasets)

    elapsed_time = time.time() - start_time
    run_report.full_train_duration = "{:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60
    )
    logger.info(
        "Full training complete in {}".format(run_report.full_train_duration))

    if args.tsne: #todo not relevant
        start_time = time.time()
        tsne_decomp, labels_tsne = run_tsne(qsn, datasets) #Returns tsne decomposition and correctly ordered labels
        elapsed_time = time.time() - start_time
        run_report.tsne_duration = "{:.0f}m {:.0f}s".format(
            elapsed_time // 60, elapsed_time % 60
        )
        logger.info(
            "TSNE complete in {}".format(run_report.tsne_duration))

        # Create frames to animate the process
        frames = make_animation_from_data(tsne_decomp, labels_tsne, active_learning_ds.labelling_progress, ["no damage", "minor damage", "major damage", "destroyed"])
        make_animation(args.checkpoint_path, frames)

        tsne_actds_dict = {'active_learning_ds': active_learning_ds, 'tsne_decomp': tsne_decomp, 'frames': frames}
        with open(os.path.join(args.checkpoint_path,'tsne_activeds_dict.pickle'), 'wb') as handle:
            pickle.dump(tsne_actds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info("END")
    save_run_report(run_report)



if __name__ == "__main__":
    main()
