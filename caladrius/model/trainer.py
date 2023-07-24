import os
import copy
import time
import pickle
from datetime import datetime
import torch
from statistics import mode, mean, median

#change Polle
from typing import Callable, Optional
from torch.utils.data.dataloader import default_collate
from baal.utils.iterutils import map_on_tensor
from baal.utils.array_utils import stack_in_memory
from collections.abc import Sequence
import numpy as np
from baal.active.heuristics import BALD, BatchBALD
from baal.active import FileDataset, ActiveLearningDataset #todo: keep this
from baal.bayesian.dropout import MCDropoutModule
from model.WAAL import set_requires_grad, gradient_penalty, WAAL_query, pred_dis_score
import torch.nn.functional as F
import gc

from torch.optim import Adam
from torch.nn.modules import loss as nnloss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch import nn, cdist

from model.networks.inception_siamese_network import (
    get_pretrained_iv3_transforms,
    InceptionSiameseNetwork,
    InceptionSiameseShared,
)
from model.networks.light_siamese_network import (
    get_light_siamese_transforms,
    LightSiameseNetwork,
)
from model.networks.vgg_siamese_network import (
    VggSiameseNetwork,
    get_pretrained_vgg_transforms,
)

from model.networks.inception_cnn_network import InceptionCNNNetwork

from utils import create_logger, readable_float, dynamic_report_key
from model.evaluate import RollingEval

logger = create_logger(__name__)

# for debugging, to profile how long different parts of trainer take
try:
    profile  # throws an exception when profile isn't defined
except NameError:

    def profile(x):
        return x


class QuasiSiameseNetwork(object):
    def __init__(self, args):
        input_size = (args.input_size, args.input_size)

        self.run_name = args.run_name
        self.input_size = input_size
        self.lr = args.learning_rate
        self.output_type = args.output_type
        self.test_epoch = args.test_epoch
        self.freeze = args.freeze
        self.no_augment = args.no_augment
        self.augment_type = args.augment_type
        self.weighted_loss = args.weighted_loss
        self.save_all = args.save_all
        self.probability = args.probability
        self.active = args.active
        if self.active:
            self.initial_number_active = args.initial_number_active
            self.active_images_per_iteration = args.active_images_per_iteration
        if self.active == 'bald' or 'batchbald': #todo: irrelevant
            self.MC_iterations = args.MC_iterations
            self.replicate_in_memory = True #Used for more efficient computation of MC dropout; set to False if this yield memory issues
        if self.active == 'batchbald': #todo: irrelevant
            self.num_draw_batchbald = args.num_draw_batchbald
        network_architecture_class = InceptionSiameseNetwork
        network_architecture_transforms = get_pretrained_iv3_transforms
        if args.model_type == "shared":
            network_architecture_class = InceptionSiameseShared
            network_architecture_transforms = get_pretrained_iv3_transforms
        if args.model_type == "light":
            network_architecture_class = LightSiameseNetwork
            network_architecture_transforms = get_light_siamese_transforms
        if args.model_type == "after":
            network_architecture_class = InceptionCNNNetwork
            network_architecture_transforms = get_pretrained_iv3_transforms
        if args.model_type == "vgg":
            network_architecture_class = VggSiameseNetwork
            network_architecture_transforms = get_pretrained_vgg_transforms

        # define the loss measure
        if self.output_type == "regression":
            self.criterion = nnloss.MSELoss()
            self.model = network_architecture_class()
        elif self.output_type == "classification":
            self.number_classes = args.number_classes
            self.model = network_architecture_class(
                output_type=self.output_type,
                n_classes=self.number_classes,
                freeze=self.freeze,
            )
            self.criterion = nnloss.CrossEntropyLoss()

        self.transforms = {}

        if torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        for s in ("train", "validation", "test", "inference"):
            self.transforms[s] = network_architecture_transforms(
                s, self.no_augment, self.augment_type
            )

        logger.debug("Num params: {}".format(len([_ for _ in self.model.parameters()])))

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        # reduces the learning rate when loss plateaus, i.e. doesn't improve
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
        )
        # creates tracking file for tensorboard
        self.writer = SummaryWriter(args.checkpoint_path)

        self.device = args.device
        self.model_path = args.model_path
        self.prediction_path = args.prediction_path
        # self.distance_path = args.distance_path
        self.model_type = args.model_type
        self.is_statistical_model = args.statistical_model
        self.is_neural_model = args.neural_model
        self.log_step = args.log_step
        self.log_step = args.log_step



    @profile
    def define_loss(self, dataset):
        if self.output_type == "regression":
            self.criterion = nnloss.MSELoss()
        else:
            if self.weighted_loss:
                num_samples = len(dataset)

                # distribution of classes in the dataset
                label_to_count = {n: 0 for n in range(self.number_classes)}
                for idx in list(range(num_samples)):
                    label = dataset.load_datapoint(idx)[-1]
                    label_to_count[label] += 1

                label_percentage = {
                    l: label_to_count[l] / num_samples for l in label_to_count.keys()
                }
                median_perc = median(list(label_percentage.values()))
                class_weights = [
                    median_perc / label_percentage[c] if label_percentage[c] != 0 else 0
                    for c in range(self.number_classes)
                ]
                weights = torch.FloatTensor(class_weights).to(self.device)

            else:
                weights = None
            self.criterion = nnloss.CrossEntropyLoss(weight=weights)

    def get_random_output_values(self, output_shape):
        return torch.rand(output_shape)

    def get_average_output_values(self, output_size, average_label):
        if self.output_type == "regression":
            outputs = torch.ones(output_size) * average_label
        elif self.output_type == "classification":
            average_label_tensor = torch.zeros(self.number_classes)
            average_label_tensor[average_label] = 1
            outputs = average_label_tensor.repeat(output_size[0], 1)
        return outputs

    def calculate_average_label(self, train_set):
        list_of_labels = []
        for _, _, _, label in train_set:
            list_of_labels.extend([label])
        if self.output_type == "regression":
            average_label = mean(list_of_labels)
        elif self.output_type == "classification":
            # use mode as average. mode=number that occurs most often in the list
            average_label = int(mode(list_of_labels))
        return average_label

    def create_prediction_file(self, phase, epoch, active_iteration = False, validation_pre_trained = False):

        if self.probability:
            if validation_pre_trained:
                prediction_file_name = "valid_pre_trained-{}-split_{}-epoch_{:03d}-model_probability-predictions.txt".format(
                    self.run_name, phase, epoch
                )
            elif str(active_iteration) != str(False): #Done with str() such that 0 != False
                prediction_file_name = "active_iteration_{}-{}-split_{}-epoch_{:03d}-model_probability-predictions.txt".format(
                    active_iteration,self.run_name, phase, epoch
                )
            else:
                prediction_file_name = "{}-split_{}-epoch_{:03d}-model_probability-predictions.txt".format(
                    self.run_name, phase, epoch
                )
            prediction_file_path = os.path.join(
                self.prediction_path, prediction_file_name
            )
            return open(prediction_file_path, "wb")
        else:
            if validation_pre_trained:
                prediction_file_name = "valid_pre_trained-{}-split_{}-epoch_{:03d}-model_probability-predictions.txt".format(
                    self.run_name, phase, epoch
                )
            elif str(active_iteration) != str(False):
                prediction_file_name = "active_iteration_{}-{}-split_{}-epoch_{:03d}-model_{}-predictions.txt".format(
                    active_iteration,self.run_name, phase, epoch, self.model_type
                )
            else:
                prediction_file_name = "{}-split_{}-epoch_{:03d}-model_{}-predictions.txt".format(
                    self.run_name, phase, epoch, self.model_type
                )
            prediction_file_path = os.path.join(
                self.prediction_path, prediction_file_name
            )
            prediction_file = open(prediction_file_path, "w+")
            prediction_file.write("filename label prediction\n")
            return prediction_file

    @profile
    def get_outputs_preds(
        self, image1, image2, random_target_shape, average_target_size
    ):
        if self.probability:
            # CHANGE SANNE
            # ORIGINAL: outputs = nn.functional.softmax(self.model(image1, image2), dim=1).squeeze()
            outputs, intermediate_results = self.model(image1, image2)
            outputs = nn.functional.softmax(outputs, dim=1).squeeze()
        elif self.is_neural_model:
            # outputs = self.model(image1, image2).squeeze() #original
            # #CHANGE SANNE
            outputs, intermediate_results = self.model(image1, image2)
            outputs = outputs.squeeze()
        elif self.model_type == "random":
            output_shape = (
                random_target_shape
                if self.output_type == "regression"
                else (random_target_shape[0], self.number_classes)
            )
            outputs = self.get_random_output_values(output_shape)
        elif self.model_type == "average":
            outputs = self.get_average_output_values(
                average_target_size, self.average_label
            )

        outputs = outputs.to(self.device)
        if self.output_type == "classification":
            _, preds = torch.max(outputs, 1)
        else:
            preds = outputs.clamp(0, 1)

        return outputs, preds, intermediate_results #CHANGE SANNE #Todo: intermediate results part probably not relevant

    @profile
    def run_epoch(
        self,
        epoch,
        loader,
        phase="train",
        train_set=None,
        selection_metric="recall_micro",
        active_selection_iter = False,
        validation_pre_trained = False,
    ):
        """
        Run one epoch of the model
        Args:
            epoch (int): which epoch this is
            loader: loader object with data
            phase (str): which phase to run epoch for. 'train', 'validation' or 'test'

        Returns:
            epoch_loss (float): loss of this epoch
            epoch_score (float): micro recall of this epoch
        """
        assert phase in ("train", "validation", "test", "train_valid") #train validation added for the case we want to use the train data but only for for validation, which is used with WAAL

        self.model = self.model.to(self.device)

        self.model.eval()
        if phase == "train":
            self.model.train()  # Set model to training mode

        rolling_eval = RollingEval(self.output_type)
        if phase ==  "train_valid": #In this case, name of prediction file must be train, but rest should be performed as if doing validation
            prediction_file = self.create_prediction_file("train", epoch, active_selection_iter, validation_pre_trained)
            # phase = "validation"
        else:
            prediction_file = self.create_prediction_file(phase, epoch, active_selection_iter, validation_pre_trained)
        # distance_file = self.create_distance_file(phase, epoch)

        if self.model_type == "average":
            self.average_label = self.calculate_average_label(train_set)

        if self.probability:
            output_probability_list = []

        for idx, (filename, image1, image2, labels) in enumerate(loader, 1):
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)
            if self.output_type == "regression":
                labels = labels.float()
            else:
                labels = labels.long()
            labels = labels.to(self.device)

            if phase == "train":
                # zero the parameter gradients
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs, preds, intermediate_results = self.get_outputs_preds( #CHANGE SANNE
                    image1, image2, labels.shape, labels.shape
                )
                loss = self.criterion(outputs, labels)
                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

                if self.probability:
                    output_probability_list.extend(outputs.tolist())
                    # distance_file.writelines(
                    #     [
                    #         "{} {} \n".format(*line)
                    #         for line in zip(
                    #             filename,
                    #             outputs,#outputs,
                    #         )
                    #     ]
                    # )
                else:
                    #CHANGE SANNE
                    # distance_file.writelines(
                    #     [
                    #         "{} {} \n".format(*line)
                    #         for line in zip(
                    #             filename,
                    #             intermediate_results,#.items(),
                    #         )
                    #     ]
                    # )  #Probably not needed as it seems coreset
                    prediction_file.writelines(
                        [
                            "{} {} {}\n".format(*line)
                            for line in zip(
                                filename,
                                labels.view(-1).tolist(),
                                preds.view(-1).tolist(),
                            )
                        ]
                    )

                batch_loss = loss.item()
                batch_score = rolling_eval.add(labels, preds, batch_loss)

            if idx % self.log_step == 0:
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: Loss: {:.4f} Accuracy: {:.4f} Correct: {:d} Total: {:d}".format(
                        epoch,
                        phase,
                        idx,
                        len(loader),
                        batch_loss,
                        batch_score[0],
                        batch_score[1],
                        batch_score[2],
                    )
                )
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: (Micro) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                        epoch,
                        phase,
                        idx,
                        len(loader),
                        batch_score[3][0],
                        batch_score[3][1],
                        batch_score[3][2],
                    )
                )
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: (Macro) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                        epoch,
                        phase,
                        idx,
                        len(loader),
                        batch_score[4][0],
                        batch_score[4][1],
                        batch_score[4][2],
                    )
                )
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: (Weighted) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                        epoch,
                        phase,
                        idx,
                        len(loader),
                        batch_score[5][0],
                        batch_score[5][1],
                        batch_score[5][2],
                    )
                )

        epoch_loss = rolling_eval.loss()
        epoch_score = rolling_eval.score()

        second_index_key, first_index_key = selection_metric.split("_")

        first_index = {"micro": 3, "macro": 4, "weighted": 5}
        second_index = {"precision": 0, "recall": 1, "f1": 2}

        epoch_main_metric = epoch_score[first_index[first_index_key]][
            second_index[second_index_key]
        ]

        if self.probability:
            #Change Sanne
            pickle.dump(output_probability_list, prediction_file)
            #prediction_file.writelines(output_probability_list)


        prediction_file.close()

        logger.info(
            "Epoch {:03d} Phase: {:10s}: Loss: {:.4f} Accuracy: {:.4f} Correct: {:d} Total: {:d}".format(
                epoch, phase, epoch_loss, epoch_score[0], epoch_score[1], epoch_score[2]
            )
        )
        logger.info(
            "Epoch {:03d} Phase: {:10s}: (Micro) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                epoch, phase, epoch_score[3][0], epoch_score[3][1], epoch_score[3][2]
            )
        )
        logger.info(
            "Epoch {:03d} Phase: {:10s}: (Macro) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                epoch, phase, epoch_score[4][0], epoch_score[4][1], epoch_score[4][2]
            )
        )
        logger.info(
            "Epoch {:03d} Phase: {:10s}: (Weighted) Precision: {:.4f} Recall: {:.4f} F1: {:.4f}".format(
                epoch, phase, epoch_score[5][0], epoch_score[5][1], epoch_score[5][2]
            )
        )

        return epoch_loss, epoch_main_metric

    @profile
    def train(self, run_report, datasets, number_of_epochs, selection_metric, active_learning_ds = False, last_query = False, active_iteration = False):
        """
        Train the model
        Args:
            run_report (dict): configuration parameters for reporting training statistics
            datasets: DataSet object with datasets loaded
            number_of_epochs (int): number of epochs to be run
            active: (added polle): if active learning is needed, this specifies the query method

        Returns:
            run_report (dict): configuration parameters for training with training statistics
        """
        active_selection_iter = active_iteration + 1 #If i = 0, active selection number 1 is performed etc
        if str(active_iteration) == str(0): #No AL has been performed yet, just some random initial selection; str as False == 0 in if statement
            active_iteration = 'random initial dataset'
        train_set, train_loader = datasets.load("train")
        validation_set, validation_loader = datasets.load("validation")
        testrunning_set, testrunning_loader = datasets.load("test")

        if self.active:
            if not active_learning_ds: #in this case it is the first iteration and active_learning_ds must still be created
                active_learning_ds = self.initialize_active_learning_dataset(datasets, seedselection=555)
            # print the number of currently labelled images
            print(f"Num. labeled: {len(active_learning_ds)}/{len(train_set)}")
            # Included to extract the true labels from after having chosen images to label
            train_set_for_active = train_set
            # If active learning is enabled, this should ensure the right selection of files in new dataloader
            active_idx = active_learning_ds.labelled
            # create new train files containing only the labelled instances
            train_set, train_loader = datasets.load("train", active_idx)

        if self.active == 'WAAL': #Todo: irrelevant
            # remove optimizer to save memory; could be left out when sufficient memory available
            if active_selection_iter == 1:
                del self.optimizer
                del self.lr_scheduler
            #WAAL needs a different type of training
            #first initializing some reporting things which are included in regular training scheme of caladrius
            start_time = time.time()
            run_report.train_start_time = (
                datetime.utcnow().replace(microsecond=0).isoformat()
            )
            run_report.train_loss = []
            run_report.train_score = []
            run_report.validation_loss = []
            run_report.validation_score = []

            # Some WAAL-train specific initialization for unlabelled pool
            active_idx_pool = np.invert(active_idx)
            pool_set, pool_loader = datasets.load("train", active_idx_pool)

            #Run WAAL adversarial training
            run_report = self.waal_trainer(alpha=0.01, n_epoch=number_of_epochs, active_learning_ds=active_learning_ds,
                              train_loader=train_loader, pool_loader=pool_loader, validation_loader=validation_loader,
                                           run_report=run_report, selection_metric=selection_metric,
                                           active_selection_iter=active_selection_iter)
            time_elapsed = time.time()-start_time
            run_report.train_duration = "{:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
            logger.info(
                "Training active iteration {} complete in {}".format(active_iteration, run_report.train_duration))
        else: #regular training
            self.define_loss(train_set)
            best_validation_score, best_model_wts = (
                0.0,
                copy.deepcopy(self.model.state_dict()),
            )

            start_time = time.time()
            run_report.train_start_time = (
                datetime.utcnow().replace(microsecond=0).isoformat()
            )
            run_report.train_loss = []
            run_report.train_score = []
            run_report.validation_loss = []
            run_report.validation_score = []

            # train network for number_of_epochs epochs
            for epoch in range(1, number_of_epochs + 1):

                #For large epoch sizes, learning rate may become too small (this was a problem for large batch sizes
                # Therefore, after each 20 epochs, the learning rate is reset
                if self.active:
                    # If-statement below can be changed to an argument if this resetting is used in practice!
                    # At this moment, it resets it before starting epoch 21,41,61 etc, as well as for 1 but there it is irrelevent
                    # Note: for testing influence of number of epochs, this may be relevant as well but is not tested due to time restraints
                    if epoch % 20 == 1: #Todo: could be relevant, but needs more testing
                        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
                        self.lr_scheduler = ReduceLROnPlateau(
                            self.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
                        )
                        print("warm restart learning rate")
                train_loss, train_score = self.run_epoch(
                    epoch, train_loader, phase="train", selection_metric=selection_metric, active_selection_iter=active_selection_iter-1
                )
                run_report.train_loss.append(readable_float(train_loss))
                run_report.train_score.append(readable_float(train_score))

                # eval on validation
                validation_loss, validation_score = self.run_epoch(
                    epoch,
                    validation_loader,
                    phase="validation",
                    selection_metric=selection_metric,
                    active_selection_iter=active_selection_iter-1,
                )
                run_report.validation_loss.append(readable_float(validation_loss))
                run_report.validation_score.append(readable_float(validation_score))

                # used for Tensorboard
                self.writer.add_scalar("Train/Loss", train_loss, epoch)
                self.writer.add_scalar("Train/Score", train_score, epoch)
                self.writer.add_scalar("Validation/Loss", validation_loss, epoch)
                self.writer.add_scalar("Validation/Score", validation_score, epoch)

                if self.test_epoch:
                    # eval on test while training
                    testrunning_loss, testrunning_accuracy = self.run_epoch(
                        epoch,
                        testrunning_loader,
                        phase="test",  # might have to do phase=val here?
                        selection_metric=selection_metric,
                        active_selection_iter=active_selection_iter-1,
                    )
                    run_report.testrunning_loss.append(readable_float(testrunning_loss))
                    run_report.testrunning_accuracy.append(
                        readable_float(testrunning_accuracy)
                    )
                    self.writer.add_scalar("Testrunning/Loss", testrunning_loss, epoch)
                    self.writer.add_scalar(
                        "Testrunning/Accuracy", testrunning_accuracy, epoch
                    )

                self.lr_scheduler.step(validation_loss)

                if validation_score > best_validation_score: #Todo: Probably this needs to be excluded when using AL in practice, since no validation set is available
                    best_validation_score = validation_score
                    best_model_wts = copy.deepcopy(self.model.state_dict())

                    logger.info(
                        "Epoch {:03d} Checkpoint: Saving to {}".format(
                            epoch, self.model_path
                        )
                    )
                    torch.save(best_model_wts, self.model_path)


            time_elapsed = time.time() - start_time
            run_report.train_end_time = datetime.utcnow().replace(microsecond=0).isoformat()

            run_report.train_duration = "{:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
            if active_iteration:
                logger.info("Training active iteration {} complete in {}".format(active_iteration, run_report.train_duration))
            else:
                logger.info(
                    "Training complete in {}".format(run_report.train_duration))
            if active_iteration:
                logger.info("Best validation score in active iteration {}: {:4f}.".format(active_iteration,best_validation_score))
            else:
                logger.info("Best validation score: {:4f}.".format(best_validation_score))

        if self.active and not last_query:
            # Set which imaages are not yet labelled
            active_idx_pool = np.invert(active_idx)
            # create pool from which instances can be selected; maybe this should be changed to inference in a practical situation such that labels are removed
            pool_set, pool_loader = datasets.load("train",active_idx_pool,active_inference = True) #todo: check if it can be deleted with random to prevent unnecessary loading
            if self.active == 'bald' or self.active == 'batchbald': #todo: not relevant
                active_learning_ds, run_report = self.label_images_bayesian_al(active_learning_ds,
                                                                                       pool_loader,
                                                                                       train_set_for_active,
                                                                                       active_selection_iter,
                                                                                       run_report)
            elif self.active == 'WAAL': #todo: irrelevant
                print('WAAL query')
                start_time = time.time()
                #Set below to true to find out about the mean and std of dis predictions on train set
                if True:
                    dis_score = pred_dis_score(qsn=self, loader=train_loader)
                    print('dis scores labelled')
                    print(dis_score[torch.isfinite(dis_score)].mean())
                    print(dis_score[torch.isfinite(dis_score)].std())
                    print(dis_score.min())
                    print(dis_score.max())
                query_pool_images = WAAL_query(qsn=self, query_num=self.active_images_per_iteration, pool_loader=pool_loader) #Todo: work on this part from here; check if self can be given as argument like this; if so remove imagesperiteration part
                time_elapsed = time.time() - start_time
                run_report.WAAL_query_duration = "{:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
                logger.info(
                    "Query with WAAL in active iteration {} completed in {}".format(active_selection_iter,
                                                                                                 run_report.WAAL_query_duration))
                # Label selected images
                self.label_selected_images(query_pool_images, train_set_for_active, active_learning_ds)


            else: #todo: probably take out of else statement
                logger.info("random selection in active iteration {} completed".format(active_selection_iter))
                #Randomly select images to label
                random_chosen = np.random.choice(active_learning_ds.n_unlabelled, self.active_images_per_iteration, replace = False)
                #Label selected images
                self.label_selected_images(random_chosen, train_set_for_active, active_learning_ds)

        if self.active: #return some more parts which are needed
            # Below is to reset learning rate so it will not keep decreasing when new images are added
            if not self.active=='WAAL':
                self.optimizer = Adam(self.model.parameters(), lr=self.lr)
                self.lr_scheduler = ReduceLROnPlateau(
                    self.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
                )
            return run_report, active_learning_ds
        return run_report

    def label_selected_images(self, chosen_images, train_set_for_active, active_learning_ds):
        """
        Labels the images which were selected by the model.
        Note: At this moment this function is not robust to changes in dataset type: .find() is used with datapoints to
        extract names and labels in dataset, which could stop working when the txt files storing these change
        :param chosen_images: Numpy array containing indices of which images the model selected to be labelled. Important
        to note is that these indices relate to the unlabelled pool, not the full dataset (those indices are set in this
        function)
        :param train_set_for_active: Full CaladriusDataset object in which the all used images, both from the labelled
        and unlabelled pool, can be found
        :param active_learning_ds: ActiveLearningDataset object storing which images are labelled previously and when.
        The function labels the chosen images within this object
        """
        #Extract the right oracle indices from chosen images -> images are chosen from the unlabelled pool, but when
        #labelling we want to know it's position in the full training set provided to label that image
        oracle_indices = active_learning_ds._pool_to_oracle_index(chosen_images)
        #Extract the right labels  Todo in practice: Here a part should be added to add labels manually by humans
        labels = [train_set_for_active.datapoints[idx][(train_set_for_active.datapoints[idx].find('.') + 5):]
                  for idx in oracle_indices]
        #Add the labels in active_learning_ds dataset and set the images to being labelled Todo in practice: Actually add these labels in a file to be extracted by train loader later
        active_learning_ds.label(chosen_images, labels)
        #Store labelling progress which can be used when making t-SNE plots
        active_learning_ds.labelling_progress += active_learning_ds._labelled.astype(np.uint16)  # For the tsne plot

    def label_images_bayesian_al(self, active_learning_ds, pool_loader, train_set_for_active, active_selection_iter,
                                 run_report): #todo: irrelevant
        """
        Uses Bayesian query and adds selected images to dataset
        """
        if self.active == 'bald':
            heuristic = BALD()
        elif self.active == 'batchbald':
            heuristic = BatchBALD(num_samples=self.active_images_per_iteration,
                                  num_draw=self.num_draw_batchbald)  # , #advised numdraw: 40000 // k,
            # See batchbald function. However, this advised numdraw leads to too large computational costs (in some cases even memory errors
        start_time = time.time()
        self.model = MCDropoutModule(self.model)  # make the model a bayesian MC dropout model
        use_cuda = torch.cuda.is_available()
        predictions = self.predict_on_dataset_bal(pool_loader, batch_size=pool_loader.batch_size,
                                                  iterations=self.MC_iterations,
                                                  use_cuda=use_cuda, verbose=False,
                                                  workers=pool_loader.num_workers)  # make predictions using bayesian active learning (bal) functions based on package BAAL
        # Todo: Specify function names
        time_elapsed = time.time() - start_time
        self.model = self.model.unpatch()  # Reset model to standard (non MC dropout) version
        run_report.BCNN_pred_duration = "{:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
        logger.info("Prediction with BCNN on pool in active iteration {} completed in {}".format(active_selection_iter,
                                                                                                 run_report.BCNN_pred_duration))

        start_time = time.time()
        top_uncertainty = heuristic(predictions)[
                          :self.active_images_per_iteration]  # Select the ones with the highest score; Note: SoftMax function is implemented within this function (in the wrapper part of heuristics.py in BAAL)

        time_elapsed = time.time() - start_time
        run_report.heuristic_uncertainty_duration = "{:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
        logger.info("Selecting samples with uncertainty heuristic in active iteration {} complete in {}".format(
            active_selection_iter, run_report.heuristic_uncertainty_duration))
        # Label selected images
        self.label_selected_images(top_uncertainty, train_set_for_active, active_learning_ds)
        return active_learning_ds, run_report

    def initialize_active_learning_dataset(self, datasets, seedselection):
        """
        Creates the active_learning_ds object to keep track of active learning and performs and labels initial labelled
        pool
        Note: At this moment this function is not robust to changes in dataset type: .find() is used with datapoints to
        extract names and labels in dataset, which could stop working when the txt files storing these change
        :param datasets: A Datasets object
        :param seedselection: To enable setting a seed in research setting. Either integer specifying the seed, or False
        :return: active_learning_ds: ActiveLearningDataset object storing which images are labelled
        """
        train_set, train_loader = datasets.load("train")
        # To contruct train set similar to BAAL framework:
        train_active_names = [s[:(s.find(".") + 4)] for s in train_set.datapoints]
        # Define dataset in BAAL framework type
        train_active_dataset = FileDataset(train_active_names, [-1] * len(train_active_names), train_set.transforms)
        # Define BAAL ActiveLearningDataset, later used to label instances, and keep track of the labelling
        active_learning_ds = ActiveLearningDataset(train_active_dataset, pool_specifics={
            'transform': train_set.transforms})
        # To always select the same starting data points NOTE: When doing multiple runs, change this
        if seedselection:
            np.random.seed(seedselection)
        # labelling initial random datapoints
        train_idxs = np.random.permutation(np.arange(len(train_set.datapoints)))[:self.initial_number_active].tolist()
        # extracts the labels by reading in string from 5 spaces next to the '.'. Todo in practice: Here a part should be added to add labels manually by humans
        labels = [train_set.datapoints[idx][(train_set.datapoints[idx].find('.') + 5):] for idx in train_idxs]
        # setting these images to labelled; Todo in practice: Actually add these labels in a file to be extracted by train loader later
        active_learning_ds.label(train_idxs, labels)
        # for TSNE plot, as here the first labelled images must bethe highest number for the visualization
        active_learning_ds.labelling_progress = active_learning_ds._labelled.copy().astype(
            np.uint16)
        return active_learning_ds

    def _stack_preds_bal(self,out):
        """
        Needed for Bayesian active learning
        """
        if isinstance(out[0], Sequence):
            out = [torch.stack(ts, dim=-1) for ts in zip(*out)]
        else:
            out = torch.stack(out, dim=-1)
        return out
    def predict_on_batch_bal(self, image1, image2, use_cuda, iterations=1): #added based on BAAL function
        """
        Get the model's MC prediction on a batch of images simultaneously. Based on
        https://baal.readthedocs.io/en/latest/, adjusted to work with Caladrius

        Args:
            data (Tensor): The model input.
            iterations (int): Number of MC prediction to perform for each image.
            cuda (bool): Use CUDA or not.

        Returns:
            Tensor, the loss computed from the criterion.
                    shape = {batch_size, nclass, n_iteration}.

        Raises:
            Raises RuntimeError if CUDA runs out of memory during data replication.
        """
        with torch.no_grad():
            if self.replicate_in_memory:
                image1 = map_on_tensor(lambda d: stack_in_memory(d, iterations), image1)
                image2 = map_on_tensor(lambda d: stack_in_memory(d, iterations), image2)
                try:
                    if use_cuda:
                        out = self.model(image1.cuda(),image2.cuda())
                    else:
                        out = self.model(image1, image2)
                    out = out[0] #With changes Sanne, the output also contains inputs of classification layer, which should not be used here. Potentially this can be changed by removing Sanne's codes, but might still be useful at some point. For now just select the first output, which is the one needed
                except RuntimeError as e:
                    raise RuntimeError(
                        """CUDA ran out of memory while BaaL tried to replicate data. See the exception above.
                    Use `replicate_in_memory=False` in order to reduce the memory requirements.
                    Note that there will be some speed trade-offs"""
                    ) from e
                out = map_on_tensor(lambda o: o.view([iterations, -1, *o.size()[1:]]), out)
                out = map_on_tensor(lambda o: o.permute(1, 2, *range(3, o.ndimension()), 0), out)
            else:
                if use_cuda:
                    out = [self.model(image1.cuda(),image2.cuda())[0] for _ in range(iterations)]
                else:
                    out = [self.model(image1, image2)[0] for _ in range(iterations)]
                out = self._stack_preds_bal(out)
            return out

    def predict_on_dataset_generator_bal( #added based on BAAL function
        self,
        dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 8,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time. Based on https://baal.readthedocs.io/en/latest/,
        adjusted to work with Caladrius

        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to display progress

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Generators [batch_size, n_classes, ..., n_iterations].
        """
        if len(dataset) == 0:
            return None

        # log.info("Start Predict", dataset=len(dataset))
        collate_fn = collate_fn or default_collate
        # loader = DataLoader(dataset, batch_size, False, num_workers=workers, collate_fn=collate_fn)
        loader = dataset
        # if verbose:
        #     loader = tqdm(loader, total=len(loader), file=sys.stdout)
        # for idx, (data, _) in enumerate(loader):
        for idx, (filename, image1, image2, labels) in enumerate(loader):  #This loader is of the caladrius loader type;
            # To keep this data structure, this predict function cannot be directly used from BAAL package. Loader from
            # BAAL package is meant for single picture which is why it cannot be easily used, while the same holds for
            # The prediction part itself
            pred = self.predict_on_batch_bal(image1,image2,use_cuda,iterations)
            pred = map_on_tensor(lambda x: x.detach(), pred)
            if half:
                pred = map_on_tensor(lambda x: x.half(), pred)
            yield map_on_tensor(lambda x: x.cpu().numpy(), pred)

    def predict_on_dataset_bal( #added based on BAAL function
        self,
        dataset,
        batch_size: int,
        iterations: int,
        use_cuda: bool,
        workers: int = 8,
        collate_fn: Optional[Callable] = None,
        half=False,
        verbose=True,
    ):
        """
        Use the model to predict on a dataset `iterations` time. Based on https://baal.readthedocs.io/en/latest/,
        adjusted to work with Caladrius

        Args:
            dataset (Dataset): Dataset to predict on.
            batch_size (int):  Batch size to use during prediction.
            iterations (int): Number of iterations per sample.
            use_cuda (bool): Use CUDA or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            half (bool): If True use half precision.
            verbose (bool): If True use tqdm to show progress.

        Notes:
            The "batch" is made of `batch_size` * `iterations` samples.

        Returns:
            Array [n_samples, n_outputs, ..., n_iterations].
        """
        preds = list(
            self.predict_on_dataset_generator_bal(
                dataset=dataset,
                batch_size=batch_size,
                iterations=iterations,
                use_cuda=use_cuda,
                workers=workers,
                collate_fn=collate_fn,
                half=half,
                verbose=verbose,
            )
        )

        if len(preds) > 0 and not isinstance(preds[0], Sequence):
            # Is an Array or a Tensor
            return np.vstack(preds)
        return [np.vstack(pr) for pr in zip(*preds)]

    def test(self, run_report, datasets, selection_metric):
        """
        Test the model
        Args:
            run_report (dict): configuration parameters for reporting testing statistics
            datasets: DataSet object with datasets loaded

        Returns:
            run_report (dict): configuration parameters for testing with testing statistics
        """
        if self.is_statistical_model:
            train_set, _ = datasets.load("train")
        else:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
            if self.save_all:
                torch.save(self.model, "{}_full.pkl".format(self.model_path[:-4]))
        test_set, test_loader = datasets.load("test")
        start_time = time.time()
        run_report[
            dynamic_report_key(
                "test_start_time", self.model_type, self.is_statistical_model
            )
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())
        test_loss, test_score = self.run_epoch(
            1,
            test_loader,
            phase="test",
            train_set=train_set if self.is_statistical_model else None,
            selection_metric=selection_metric,
        )
        run_report[
            dynamic_report_key("test_loss", self.model_type, self.is_statistical_model)
        ] = readable_float(test_loss)
        run_report[
            dynamic_report_key("test_score", self.model_type, self.is_statistical_model)
        ] = readable_float(test_score)
        time_elapsed = time.time() - start_time
        run_report[
            dynamic_report_key(
                "test_end_time", self.model_type, self.is_statistical_model
            )
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())

        run_report[
            dynamic_report_key(
                "test_duration", self.model_type, self.is_statistical_model
            )
        ] = "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)

        run_report[
            dynamic_report_key("test", self.model_type, self.is_statistical_model)
        ] = True

        logger.info(
            "Testing complete in {}".format(
                run_report[
                    dynamic_report_key(
                        "test_duration", self.model_type, self.is_statistical_model
                    )
                ]
            )
        )
        return run_report

    def inference(self, datasets):
        """
        Uses the model for inference
        Args:
            datasets: DataSet object with datasets loaded
        """
        if self.is_statistical_model:
            train_set, _ = datasets.load("train")
        else:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        inference_set, inference_loader = datasets.load("inference")
        start_time = time.time()

        self.model = self.model.to(self.device)

        self.model.eval()

        prediction_file = self.create_prediction_file("inference", 1)
        prediction_file.write("filename prediction\n")

        if self.model_type == "average":
            self.average_label = self.calculate_average_label(train_set)

        for idx, (filename, image1, image2) in enumerate(inference_loader, 1):
            image1 = image1.to(self.device)
            image2 = image2.to(self.device)

            outputs, preds = self.get_outputs_preds(
                image1, image2, image1.shape, [image1.shape[0]]
            )

            prediction_file.writelines(
                [
                    "{} {}\n".format(*line)
                    for line in zip(filename, preds.view(-1).tolist())
                ]
            )

        time_elapsed = time.time() - start_time

        logger.info(
            "Inference complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def validation_pre_trained(self, run_report, datasets, selection_metric):
        """
        Test the pre-trained model on validation set
        Args:
            run_report (dict): configuration parameters for reporting testing statistics
            datasets: DataSet object with datasets loaded

        Returns:
            run_report (dict): configuration parameters for testing with testing statistics
        """

        validation_set, validation_loader = datasets.load("validation")
        start_time = time.time()
        run_report[
            dynamic_report_key(
                "test_start_time", self.model_type, self.is_statistical_model
            )
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())
        validation_loss, validation_score = self.run_epoch(
            1,
            validation_loader,
            phase="validation",
            train_set=train_set if self.is_statistical_model else None,
            selection_metric=selection_metric,
            validation_pre_trained = True,
        )
        run_report[
            dynamic_report_key("validation_loss", self.model_type, self.is_statistical_model)
        ] = readable_float(validation_loss)
        run_report[
            dynamic_report_key("validation_score", self.model_type, self.is_statistical_model)
        ] = readable_float(validation_score)
        time_elapsed = time.time() - start_time
        run_report[
            dynamic_report_key(
                "validation_end_time", self.model_type, self.is_statistical_model
            )
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())

        run_report[
            dynamic_report_key(
                "validation_duration", self.model_type, self.is_statistical_model
            )
        ] = "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)

        run_report[
            dynamic_report_key("validation", self.model_type, self.is_statistical_model)
        ] = True

        logger.info(
            "Validating complete in {}".format(
                run_report[
                    dynamic_report_key(
                        "validation_duration", self.model_type, self.is_statistical_model
                    )
                ]
            )
        )
        return run_report

    def waal_trainer(self, alpha, n_epoch, active_learning_ds, train_loader, pool_loader, validation_loader, run_report,
                     selection_metric, active_selection_iter):

        """
        Only training samples with labeled and unlabeled data-set
        alpha is the trade-off between the empirical loss and error, the more interaction, the smaller alpha
        code from https://github.com/cjshui/WAAL, adjusted to work with Caladrius
        :return:
        """

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        print("[Training] labeled and unlabeled data")
        # n_epoch = self.args['n_epoch']
        # n_epoch = total_epoch

        # Send models to gpu or cpu, whichever is needed
        self.net_fea = self.net_fea.to(self.device)
        self.net_clf = self.net_clf.to(self.device)
        self.net_dis = self.net_dis.to(self.device)

        opt_fea = Adam(self.net_fea.parameters(), lr=self.lr)
        opt_clf = Adam(self.net_clf.parameters(), lr=self.lr)
        opt_dis = Adam(self.net_dis.parameters(), lr=self.lr)
        lr_scheduler_fea = ReduceLROnPlateau(opt_fea, factor=0.1, patience=10, min_lr=1e-5, verbose=True)
        lr_scheduler_clf = ReduceLROnPlateau(opt_clf, factor=0.1, patience=10, min_lr=1e-5, verbose=True)
        lr_scheduler_dis = ReduceLROnPlateau(opt_dis, factor=0.1, patience=10, min_lr=1e-5, verbose=True)

        # setting three optimizers

        # computing the unbalancing ratio, a value between [0,1], generally 0.1 - 0.5
        # gamma_ratio = active_learning_ds.n_labelled/(active_learning_ds.n_labelled+active_learning_ds.n_unlabelled)
        # gamma_ratio = active_learning_ds.n_labelled / active_learning_ds.n_unlabelled #NOte: this is the reverse of alpha in other type

        # Currently gamma=1 is used for better performance
        gamma_ratio = 1

        max_number_batches = max(len(pool_loader), len(train_loader))
        number_times_train_loader_res_per_iteration = len(pool_loader)/len(train_loader)
        print('number times train loader reset per iteration: ' + str(number_times_train_loader_res_per_iteration))
        number_times_trainset_reset = 0 #to find out how often train samples are used in each training iteration, can be used for lr scheduling
        print('number of iterations per epoch: ' + str(max_number_batches))

        #Below is to ensure training data is only used the number_wanted_epochs times; number of epochs is adjusted based onn this
        #When the required number of times training data is used is reched, require gradient is set false for the classifier
        wanted_number_of_iter = len(train_loader) * n_epoch

        number_of_epochs_needed = int(np.ceil(wanted_number_of_iter / max_number_batches))

        total_train_batched_loaded = 0
        for epoch in range(1, number_of_epochs_needed+1):
            # Set model to Identity to save memory on GPU, after training the net_fea and net_clf it is reconstructed based on these
            self.model.similarity = torch.nn.Identity()
            self.model.output = torch.nn.Identity()
            self.model.left_network = torch.nn.Identity()
            self.model.right_network = torch.nn.Identity()

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.net_fea.train()
            self.net_clf.train()
            self.net_dis.train()
            # Set below to True if batch normalization should be removed from discriminator
            if False: #Todo: check if should be enabled
                def deactivate_batchnorm(m):
                    if isinstance(m, torch.nn.BatchNorm1d):
                        m.reset_parameters()
                        m.eval()
                        with torch.no_grad():
                            m.weight.fill_(1.0)
                            m.bias.zero_()

                self.net_dis.apply(deactivate_batchnorm)


            # Total_loss = 0
            n_batch    = 0
            # acc        = 0

            iterloader_train = iter(train_loader)
            iterloader_pool = iter(pool_loader)
            # no enumeration over loader itself since loaders have different dimensions, so one needs be restarted
            for i in range(max_number_batches):
                #Try except is to make new dataloader with different distribution over batches if needed to be restarted.
                print_disr_results = False
                try:
                    (train_filename, train_image1, train_image2, train_labels) = next(iterloader_train)
                except StopIteration:
                    number_times_trainset_reset += 1 #probably base learning rate setting on this and number of epochs
                    iterloader_train = iter(train_loader)
                    (train_filename, train_image1, train_image2, train_labels) = next(iterloader_train)

                    #Since resceduling in other cases is done based on labelled set, this is done with those epochs here as well
                    # print('possibly reduce lr clf')
                    lr_scheduler_clf.step(metrics=pred_loss)
                    # print('possibly reduce lr fea')
                    lr_scheduler_fea.step(metrics=loss)
                    # print('possibly reduce lr dis')
                    lr_scheduler_dis.step(metrics=dis_loss)

                    print_disr_results = True
                    if n_epoch>99: #This is for 500 acq size: in this case use warm restart
                        if number_times_trainset_reset%20 == 0:
                            opt_fea = Adam(self.net_fea.parameters(), lr=self.lr)
                            opt_clf = Adam(self.net_clf.parameters(), lr=self.lr)
                            opt_dis = Adam(self.net_dis.parameters(), lr=self.lr)
                            lr_scheduler_fea = ReduceLROnPlateau(opt_fea, factor=0.1, patience=10, min_lr=1e-5,
                                                                 verbose=True)
                            lr_scheduler_clf = ReduceLROnPlateau(opt_clf, factor=0.1, patience=10, min_lr=1e-5,
                                                                 verbose=True)
                            lr_scheduler_dis = ReduceLROnPlateau(opt_dis, factor=0.1, patience=10, min_lr=1e-5,
                                                                 verbose=True)

                try:
                    (pool_filename, pool_image1, pool_image2, pool_labels) = next(iterloader_pool)
                except StopIteration:
                    iterloader_pool = iter(pool_loader)
                    (pool_filename, pool_image1, pool_image2, pool_labels) = next(iterloader_pool)

                n_batch += 1

                train_image1, train_image2, train_labels = train_image1.to(self.device), train_image2.to(self.device), train_labels.to(self.device)
                pool_image1, pool_image2 = pool_image1.to(self.device), pool_image2.to(self.device)

                # training feature extractor and predictor
                # adjusting total number of times a batch is loaded for train data; usig this the classifier is set to not train after these batches to prevent it from overfitting, while the feature extractor is still trained based on it.
                total_train_batched_loaded += 1
                set_requires_grad(self.net_fea,requires_grad=True)
                if not total_train_batched_loaded > wanted_number_of_iter: #Todo:set to true in other cases
                    if not total_train_batched_loaded > wanted_number_of_iter: #Todo change
                         set_requires_grad(self.net_clf,requires_grad=True)
                         # set_requires_grad(self.net_fea, requires_grad=True)
                    else:
                        set_requires_grad(self.net_clf, requires_grad=False) #todo: now feature extractor is till trained on clf loss for a long time, possibly set this loss to zero to only optimize based on wasserstein after these iterations
                        set_requires_grad(self.net_fea, requires_grad=False)
                        break
                    set_requires_grad(self.net_dis,requires_grad=False)

                    train_features = self.net_fea(train_image1, train_image2)
                    pool_features  = self.net_fea(pool_image1, pool_image2)

                    opt_fea.zero_grad()
                    opt_clf.zero_grad()

                    train_clf_pred = self.net_clf(train_features) #original was train_clf_pred, _ = self.net_clf(lb_z), but _ part seems never used, so removed for now (_ in WAAL is the return of output in earlier layer)

                    # if not total_train_batched_loaded > wanted_number_of_iter:
                        # prediction loss (deafult we use F.cross_entropy) #Similar to the loss used in Caladrius, though in Caladrius it is called through nnloss.CrossEntropyLoss
                    pred_loss = torch.mean(F.cross_entropy(train_clf_pred,train_labels.long())) #NOTE: .long() added to remove error

                    # Wasserstein loss (here is the unbalanced loss, because we used the redundant trick)
                    wassertein_distance = self.net_dis(pool_features).mean() - gamma_ratio * self.net_dis(train_features).mean()

                    # Below is to free some memory and try to make increase the capacity
                    del train_features
                    del pool_features
                    del train_clf_pred
                    del train_labels
                    self.net_clf.to('cpu')
                    self.net_dis.to('cpu')
                    gc.collect()
                    torch.cuda.empty_cache()
                    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

                    with torch.no_grad():
                        train_features = self.net_fea(train_image1, train_image2)
                        pool_features = self.net_fea(pool_image1, pool_image2)

                    self.net_clf = self.net_clf.to(self.device)
                    self.net_dis = self.net_dis.to(self.device)

                    gp = gradient_penalty(self.net_dis, pool_features, train_features)

                    loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
                    # for CIFAR10 the gradient penality is 5 #todo when implementing: check this penalty
                    # for SVHN the gradient penality is 2

                    loss.backward()
                    opt_fea.step()
                    # opt_fea.step()
                    if True:#not total_train_batched_loaded > wanted_number_of_iter: #Todo:change
                        opt_clf.step() #Here only pred_loss is used since the other terms do not involve net_clf in computations (they are based on net_dis)


                # Then the second step, training discriminator

                set_requires_grad(self.net_fea, requires_grad=False)
                set_requires_grad(self.net_clf, requires_grad=False)
                set_requires_grad(self.net_dis, requires_grad=True)


                with torch.no_grad():
                    train_features = self.net_fea(train_image1, train_image2)
                    pool_features = self.net_fea(pool_image1, pool_image2)

                for _ in range(1):

                    # gradient ascent for multiple times like GANS training

                    gp = gradient_penalty(self.net_dis, pool_features, train_features)

                    wassertein_distance = self.net_dis(pool_features).mean() - gamma_ratio * self.net_dis(train_features).mean()

                    dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

                    opt_dis.zero_grad()
                    dis_loss.backward()
                    opt_dis.step()

            #For each epoch, make predictions for both train and validation set.
            #First reconstruct full QSN model based on the trained feature extractor and classifier
            self.model.similarity = copy.deepcopy(self.net_clf.similarity)
            self.model.output = copy.deepcopy(self.net_clf.output)
            self.model.left_network = copy.deepcopy(self.net_fea.left_network)
            self.model.right_network = copy.deepcopy(self.net_fea.right_network)

            train_loss, train_score = self.run_epoch(
                epoch,
                train_loader,
                phase="train_valid", #Set to validation type as only used for creation prediction file
                selection_metric=selection_metric,
                active_selection_iter=active_selection_iter - 1,
            )
            validation_loss, validation_score = self.run_epoch(
                epoch,
                validation_loader,
                phase="validation",
                selection_metric=selection_metric,
                active_selection_iter=active_selection_iter - 1,
            )
            run_report.validation_loss.append(readable_float(validation_loss))
            run_report.validation_score.append(readable_float(validation_score))

            # used for Tensorboard
            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Score", train_score, epoch)
            self.writer.add_scalar("Validation/Loss", validation_loss, epoch)
            self.writer.add_scalar("Validation/Score", validation_score, epoch)

            # if total_train_batched_loaded > wanted_number_of_iter:
            #     break

                # prediction and computing training accuracy and empirical loss under evaluation mode
                # P = train_clf_pred.max(1)[1]
                # acc += 1.0 * (label_y == P).sum().item() / len(label_y)
                # Total_loss += loss.item()

            # Total_loss /= n_batch
            # acc        /= n_batch

            print('==========Inner epoch {:d} ========'.format(epoch))
            # print('Training Loss {:.3f}'.format(Total_loss))
            # print('Training accuracy {:.3f}'.format(acc*100))
        print("number of times train images are used in a batch: " +str(number_times_trainset_reset))
        return run_report
