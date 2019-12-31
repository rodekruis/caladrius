import os
import copy
import time
from datetime import datetime
import torch

from torch.optim import Adam
from torch.nn.modules import loss as nnloss
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from model.network import get_pretrained_iv3_transforms, SiameseNetwork
from utils import create_logger, readable_float, dynamic_report_key
from model.evaluate import RollingEval

logger = create_logger(__name__)


class QuasiSiameseNetwork(object):
    def __init__(self, args):
        input_size = (args.input_size, args.input_size)

        self.run_name = args.run_name
        self.input_size = input_size
        self.lr = args.learning_rate
        self.output_type = args.output_type

        # define the loss measure
        if self.output_type == "regression":
            self.criterion = nnloss.MSELoss()
            self.model = SiameseNetwork()
        elif self.output_type == "classification":
            self.criterion = nnloss.CrossEntropyLoss()
            self.n_classes = 4  # replace by args
            self.model = SiameseNetwork(
                output_type=self.output_type, n_classes=self.n_classes
            )

        self.transforms = {}

        if torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        for s in ("train", "validation", "test", "inference"):
            self.transforms[s] = get_pretrained_iv3_transforms(s)

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
        self.model_type = args.model_type
        self.log_step = args.log_step

    def get_random_output_values(self, output_shape):
        return torch.rand(output_shape)

    def get_average_output_values(self, output_size, average_label):
        if self.output_type == "regression":
            outputs = torch.ones(output_size) * average_label
        elif self.output_type == "classification":
            average_label_tensor = torch.zeros(self.n_classes)
            average_label_tensor[average_label] = 1
            outputs = average_label_tensor.repeat(output_size[0], 1)
        return outputs

    def calculate_average_label(self, train_set):
        sum_of_labels = 0
        for _, _, _, label in train_set:
            sum_of_labels = sum_of_labels + label
        number_of_labels = len(train_set)
        average_label = sum_of_labels / number_of_labels
        if self.output_type == "classification":
            average_label = round(average_label)
        return average_label

    def create_prediction_file(self, phase, epoch):
        prediction_file_name = "{}-split_{}-epoch_{:03d}-model_{}-predictions.txt".format(
            self.run_name, phase, epoch, self.model_type
        )
        prediction_file_path = os.path.join(self.prediction_path, prediction_file_name)
        return open(prediction_file_path, "w+")

    def get_outputs_preds(
        self, image1, image2, random_target_shape, average_target_size
    ):
        if self.model_type == "siamese":
            outputs = self.model(image1, image2).squeeze()
        elif self.model_type == "random":
            output_shape = (
                random_target_shape
                if self.output_type == "regression"
                else (random_target_shape[0], self.n_classes)
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

        return outputs, preds

    def run_epoch(self, epoch, loader, phase="train", train_set=None):
        """
        Run one epoch of the model
        Args:
            epoch (int): which epoch this is
            loader: loader object with data
            phase (str): which phase to run epoch for. 'train', 'validation' or 'test'

        Returns:
            epoch_loss (float): loss of this epoch
            epoch_accuracy (float): accuracy of this epoch
        """
        assert phase in ("train", "validation", "test")

        self.model = self.model.to(self.device)

        self.model.eval()
        if phase == "train":
            self.model.train()  # Set model to training mode

        rolling_eval = RollingEval(self.output_type)

        prediction_file = self.create_prediction_file(phase, epoch)
        prediction_file.write("filename label prediction\n")

        if self.model_type == "average":
            self.average_label = self.calculate_average_label(train_set)

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
                outputs, preds = self.get_outputs_preds(
                    image1, image2, labels.shape, labels.shape
                )
                loss = self.criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

                prediction_file.writelines(
                    [
                        "{} {} {}\n".format(*line)
                        for line in zip(
                            filename, labels.view(-1).tolist(), preds.view(-1).tolist()
                        )
                    ]
                )

                batch_loss = loss.item()
                batch_score = rolling_eval.add(labels, preds, batch_loss)

            if idx % self.log_step == 0:
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: Loss: {:.4f} Score: {:.4f}".format(
                        epoch, phase, idx, len(loader), batch_loss, batch_score
                    )
                )

        epoch_loss = rolling_eval.loss()
        epoch_score = rolling_eval.score()

        if not (phase == "train"):
            prediction_file.write(
                "Epoch {:03d} Score: {:.4f}\n".format(epoch, epoch_score)
            )
            prediction_file.close()

        logger.info(
            "Epoch {:03d} Phase: {:10s} Loss: {:.4f} Score: {:.4f}".format(
                epoch, phase, epoch_loss, epoch_score
            )
        )

        return epoch_loss, epoch_score

    def train(self, run_report, datasets, number_of_epochs):
        """
        Train the model
        Args:
            run_report (dict): configuration parameters for reporting training statistics
            datasets: DataSet object with datasets loaded
            number_of_epochs (int): number of epochs to be run

        Returns:
            run_report (dict): configuration parameters for training with training statistics
        """
        train_set, train_loader = datasets.load("train")
        validation_set, validation_loader = datasets.load("validation")

        best_accuracy, best_model_wts = 0.0, copy.deepcopy(self.model.state_dict())

        start_time = time.time()
        run_report.train_start_time = (
            datetime.utcnow().replace(microsecond=0).isoformat()
        )
        run_report.train_loss = []
        run_report.train_accuracy = []
        run_report.validation_loss = []
        run_report.validation_accuracy = []

        for epoch in range(1, number_of_epochs + 1):
            # train network
            train_loss, train_accuracy = self.run_epoch(
                epoch, train_loader, phase="train"
            )
            run_report.train_loss.append(readable_float(train_loss))
            run_report.train_accuracy.append(readable_float(train_accuracy))

            # eval on validation
            validation_loss, validation_accuracy = self.run_epoch(
                epoch, validation_loader, phase="validation"
            )
            run_report.validation_loss.append(readable_float(validation_loss))
            run_report.validation_accuracy.append(readable_float(validation_accuracy))

            self.writer.add_scalar("Train/Loss", train_loss, epoch)
            self.writer.add_scalar("Train/Accuracy", train_accuracy, epoch)
            self.writer.add_scalar("Validation/Loss", validation_loss, epoch)
            self.writer.add_scalar("Validation/Accuracy", validation_accuracy, epoch)

            self.lr_scheduler.step(validation_loss)

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
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

        logger.info("Training complete in {}".format(run_report.train_duration))

        logger.info("Best validation Accuracy: {:4f}.".format(best_accuracy))
        return run_report

    def test(self, run_report, datasets):
        """
        Test the model
        Args:
            run_report (dict): configuration parameters for reporting testing statistics
            datasets: DataSet object with datasets loaded

        Returns:
            run_report (dict): configuration parameters for testing with testing statistics
        """
        is_statistical_model = self.model_type != "siamese"
        if is_statistical_model:
            train_set, _ = datasets.load("train")
        else:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=self.device)
            )
        test_set, test_loader = datasets.load("test")
        start_time = time.time()
        run_report[
            dynamic_report_key("test_start_time", self.model_type, is_statistical_model)
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())
        test_loss, test_accuracy = self.run_epoch(
            1,
            test_loader,
            phase="test",
            train_set=train_set if is_statistical_model else None,
        )
        run_report[
            dynamic_report_key("test_loss", self.model_type, is_statistical_model)
        ] = readable_float(test_loss)
        run_report[
            dynamic_report_key("test_accuracy", self.model_type, is_statistical_model)
        ] = readable_float(test_accuracy)
        time_elapsed = time.time() - start_time
        run_report[
            dynamic_report_key("test_end_time", self.model_type, is_statistical_model)
        ] = (datetime.utcnow().replace(microsecond=0).isoformat())

        run_report[
            dynamic_report_key("test_duration", self.model_type, is_statistical_model)
        ] = "{:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)

        run_report[
            dynamic_report_key("test", self.model_type, is_statistical_model)
        ] = True

        logger.info(
            "Testing complete in {}".format(
                run_report[
                    dynamic_report_key(
                        "test_duration", self.model_type, is_statistical_model
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
        is_statistical_model = self.model_type != "siamese"
        if is_statistical_model:
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
