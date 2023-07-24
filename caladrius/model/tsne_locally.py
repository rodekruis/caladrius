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
from baal.bayesian.dropout import MCDropoutModule
from baal.utils.plot_utils import make_animation_from_data


def main(): #Runs tsne locally given activelearingds and model from training
    args = configuration()
    args.data_path = input_path
    args.output_type = 'classification'
    args.run_name = 'test_model'
    args.checkpoint_path = output_path
    args.number_of_epochs = 1
    args.batch_size = 5 #32 #On test set a high batch size leads to empty predictions and hence errors (to small data set for some batch sizes
    args.model_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\test_small\runs\models\\'  + str(args.run_name) + '.pkl'
    # args.checkpoint_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\trial\runs\output_0_file'
    # args.distance_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\matthew_all\runs\distances' #Not needed now, only possibly uncomment when we want to store something in other file, in this case rename from distance
    args.prediction_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\test_small\runs\predictions'
    args.run_report_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\test_small\runs\runs_reports\\'  + str(args.run_name) + '.txt'
    args.active = 'bald'
    args.initial_number_active = 10 #set very low for now as testing with small dataset
    args.active_images_per_iteration=10
    args.active_iterations=1
    args.MC_iterations=1
    args.tsne = False
    # args.weighted_loss = True #So that all classes are weighted equally
    # args.no_augment = True #To add augmentation
    args.number_of_workers = 1 #Lower value makes it slower, but also locally it works with bigger batches then. Default is 8
    args.pretrained_model_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\pre_trained_all_wind'
    args.pretrained_model_name = 'best_model_wts.pkl'


    file = open(r'C:\Users\PolleDankers\Downloads\tsne_activeds_dict.pickle', 'rb')
    tsne_dict = pickle.load(file)
    #load model
    args.batch_size = 30
    qsn2 = QuasiSiameseNetwork(args)
    qsn2.model.load_state_dict(torch.load(r'C:\Users\PolleDankers\Downloads\best_model_wts.pkl', map_location=torch.device('cpu')))

    args.data_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\matthew_all'
    datasets = Datasets(args, qsn2.transforms)
    tsne_decomp, labels_tsne = run_tsne(qsn2, datasets)

    make_animation_from_data(tsne_decomp, labels_tsne,
                             tsne_dict['active_learning_ds'].labelled_map.copy().astype(np.uint16),
                             ["no damage", "minor damage", "major damage", "destroyed"])

if __name__ == "__main__":
    input_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\test_small'
    output_path = r'C:\Users\PolleDankers\OneDrive - Pipple BV\Documenten\Thesis 510\data\test_small\runs\output_0_file'
    main()
