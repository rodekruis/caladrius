#Todo: Irrelevent (whole file)
import copy
import os
import sys
import argparse

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
import numpy as np
import pickle
from baal.utils.plot_utils import make_animation_from_data
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model.WAAL import caladrius_waal_extractor

#This script can be used to run Active Learning locally (to use a debugger)

def main():
    args = configuration()
    args.data_path = input_path
    args.output_type = 'classification'
    args.run_name = 'test_model'
    args.checkpoint_path = output_path
    args.number_of_epochs = 1
    args.batch_size = 5 #32 #On test set a high batch size leads to empty predictions and hence errors (to small data set for some batch sizes
    args.model_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\test_small\runs\models\\'  + str(args.run_name) + '.pkl'
    # args.checkpoint_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\trial\runs\output_0_file'
    # args.distance_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\matthew_all\runs\distances' #Not needed now, only possibly uncomment when we want to store something in other file, in this case rename from distance
    args.prediction_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\test_small\runs\predictions'
    args.run_report_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\test_small\runs\runs_reports\\'  + str(args.run_name) + '.txt'
    args.active = 'bald'
    args.initial_number_active = 10 #set very low for now as testing with small dataset
    args.active_images_per_iteration=10
    args.active_iterations=1
    args.MC_iterations=3
    args.tsne = False
    # args.weighted_loss = True #So that all classes are weighted equally
    # args.no_augment = True #To add augmentation
    args.number_of_workers = 1 #Lower value makes it slower, but also locally it works with bigger batches then. Default is 8
    args.pretrained_model_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\pre_trained_all_wind'
    args.pretrained_model_name = 'best_model_wts.pkl'
    args.tsne = True
    args.num_draw_batchbald = 7
    args.reinitialize_model = True
    args.num_epochs_last_actiter = 1

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
    qsn = QuasiSiameseNetwork(args) #Initialize empty network
    if args.pretrained_model_path:
        if torch.cuda.is_available(): #Note: This assumes the loaded model is trained on GPU and thus in cuda type, if this is not the case loaction must be mapped to cuda
            qsn.model.load_state_dict(torch.load(
            os.path.join(args.pretrained_model_path, args.pretrained_model_name)))
        else:
            qsn.model.load_state_dict(torch.load(
                os.path.join(args.pretrained_model_path, args.pretrained_model_name),map_location=torch.device('cpu')))

    if args.reinitialize_model:
        qsn_model_original = copy.deepcopy(qsn.model)

    if args.active == 'WAAL': #Divide model in three pasrts: net_fea, net_clf and net_dis
        qsn = caladrius_waal_extractor(qsn)

        #below to use layer norm instead of batch norm (also adjust parts in trainer which change batch norm)
        qsn.net_dis.similarity.bn_0 = torch.nn.LayerNorm(qsn.net_dis.similarity.layer_0.out_features)
        qsn.net_dis.similarity.bn_1 = torch.nn.LayerNorm(qsn.net_dis.similarity.layer_1.out_features)



    datasets = Datasets(args, qsn.transforms)
    if args.neural_model and not (args.test or args.inference):
        if args.active:
            if args.pretrained_model_path: #Todo: Probably remove when implementing in prctice since we do not have a labelled validation set then
                #Probably implement above to do by adding an argument for scientific mode (and use this in trainer as well)
                run_report = qsn.validation_pre_trained(run_report, datasets, args.selection_metric) #to first find the validation predictions for the pre-trained model
            active_learning_ds = False
            last_query = False #Used so that after the last query, a new model is still trained but no new query is undertaken
            num_epochs = args.number_of_epochs
            for i in range(args.active_iterations+1):
                if i == 0:
                    if args.active_images_per_iteration != args.initial_number_active: #if it is equal, epochs should be equal as well
                        if args.active == 'WAAL':
                            num_epochs = 1
                        else:
                            num_epochs = 20
                run_report, active_learning_ds = qsn.train(
                    run_report, datasets, num_epochs, args.selection_metric, active_learning_ds,last_query,i
                )
                if i == (args.active_iterations - 1): #added such that after the last query, a new model is still trained but no new query is undertaken
                    last_query = True
                    if args.num_epochs_last_actiter: #To set number of epochs to last iteration value, if this is parsed
                        num_epochs = args.num_epochs_last_actiter
                if i != args.active_iterations and args.reinitialize_model == True: #reinitialize model between active iterations
                    qsn.model = copy.deepcopy(qsn_model_original)
                    qsn.optimizer = Adam(qsn.model.parameters(), lr=qsn.lr)
                    qsn.lr_scheduler = ReduceLROnPlateau(
                        qsn.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
                    )
        else:
            if args.pretrained_model_path:
                run_report = qsn.validation_pre_trained(run_report, datasets, args.selection_metric) #to first find the validation predictions for the pre-trained model
            run_report = qsn.train(
                run_report, datasets, args.number_of_epochs, args.selection_metric
            ) #Train the empty dataset
    logger.info("Evaluating on test dataset")
    if args.active:
        #In this case, we did not have sufficient valid. data so simply pick th elast created model. NOTE: this was not
        # yet implemented when doing multiple runs on Random, BALD and BatchBALD
        best_model_wts = copy.deepcopy(qsn.model.state_dict())
        torch.save(best_model_wts, qsn.model_path)
    run_report = qsn.test(run_report, datasets, args.selection_metric) #test the model performance
    if args.inference: #Running on unlabeled data?
        logger.info("Inference started")
        qsn.inference(datasets)

    elapsed_time = time.time() - start_time
    run_report.full_train_duration = "{:.0f}m {:.0f}s".format(
        elapsed_time // 60, elapsed_time % 60
    )
    logger.info(
        "Full training complete in {}".format(run_report.full_train_duration))


    #Trial t-SNE see https://baal.readthedocs.io/en/latest/notebooks/active_learning_process/
    if args.tsne:
        start_time = time.time()
        tsne_decomp, labels_tsne = run_tsne(qsn, datasets) #Returns tsne decomposition and correctly ordered labels
        elapsed_time = time.time() - start_time
        run_report.tsne_duration = "{:.0f}m {:.0f}s".format(
            elapsed_time // 60, elapsed_time % 60
        )
        logger.info(
            "TSNE complete in {}".format(run_report.tsne_duration))

        frames = make_animation_from_data(tsne_decomp, labels_tsne, active_learning_ds.labelling_progress, ["no damage", "minor damage", "major damage", "destroyed"])
        make_animation(args.checkpoint_path, frames) #this function saves the animation, possibly leave out in online version

        tsne_actds_dict = {'active_learning_ds': active_learning_ds, 'tsne_decomp': tsne_decomp, 'frames': frames}
        with open(os.path.join(args.checkpoint_path,'tsne_activeds_dict.pickle'), 'wb') as handle:
            pickle.dump(tsne_actds_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    save_run_report(run_report)
    logger.info("END")



if __name__ == "__main__":
    input_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\test_small'
    output_path = r'C:\Users\polle\OneDrive\Documenten\Econometrie_master\Thesis_510\test_small\runs\output_0_file'
    main()

    #NOTE: if error RuntimeError: DataLoader worker (pid(s) 11160) exited unexpectedly raised: Toomuch RAM used, decrease number of epochs
