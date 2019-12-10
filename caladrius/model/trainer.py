import os
import copy
import time
import torch

from torch.optim import Adam
from torch.nn.modules import loss as nnloss
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from model.network import get_pretrained_iv3_transforms, SiameseNetwork
from utils import create_logger


logger = create_logger(__name__)


class QuasiSiameseNetwork(object):
    def __init__(self, args):
        input_size = (args.input_size, args.input_size)

        self.run_name = args.run_name
        self.input_size = input_size
        self.lr = args.learning_rate
        self.accuracy_threshold = args.accuracy_threshold

        #define the loss measure
        self.criterion = nnloss.MSELoss()

        self.transforms = {}

        self.model = SiameseNetwork()

        if torch.cuda.device_count() > 1:
            logger.info("Using {} GPUs".format(torch.cuda.device_count()))
            self.model = torch.nn.DataParallel(self.model)

        for s in ("train", "validation", "test"):
            self.transforms[s] = get_pretrained_iv3_transforms(s)

        logger.debug("Num params: {}".format(len([_ for _ in self.model.parameters()])))

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        #reduces the learning rate when loss plateaus, i.e. doesn't improve
        self.lr_scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.1, patience=10, min_lr=1e-5, verbose=True
        )
        #creates tracking file for tensorboard
        self.writer=SummaryWriter(args.checkpoint_path)

    def run_epoch(self, epoch, loader, device, predictions_path, phase="train"):
        """
        Run one epoch of the model
        Args:
            epoch (int): which epoch this is
            loader: loader object with data
            device (str): which device it is being run on. 'cpu' or 'cuda'
            predictions_path (str): path to write predictions to
            phase (str): which phase to run epoch for. 'train', 'validation' or 'test'

        Returns:
            epoch_loss (float): loss of this epoch
            epoch_accuracy (float): accuracy of this epoch
        """
        assert phase in ("train", "validation", "test")

        self.model = self.model.to(device)

        self.model.eval()
        if phase == "train":
            self.model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        running_n = 0.0

        if not (phase == "train"):
            prediction_file_name = "{}_{}_epoch_{:03d}_predictions.txt".format(
                self.run_name, phase, epoch
            )
            prediction_file_path = os.path.join(predictions_path, prediction_file_name)
            prediction_file = open(prediction_file_path, "w+")
            prediction_file.write("filename label prediction\n")

        for idx, (filename, image1, image2, labels) in enumerate(loader, 1):
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.float().to(device)

            if phase == "train":
                # zero the parameter gradients
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = self.model(image1, image2).squeeze()
                loss = self.criterion(outputs, labels)

                if not (phase == "train"):
                    prediction_file.writelines(
                        [
                            "{} {} {}\n".format(*line)
                            for line in zip(
                                filename,
                                labels.view(-1).tolist(),
                                outputs.clamp(0, 1).view(-1).tolist(),
                            )
                        ]
                    )

                if phase == "train":
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * image1.size(0)
            running_corrects += (
                (outputs - labels.data).abs().le(self.accuracy_threshold).sum()
            )
            running_n += image1.size(0)

            if idx % 1 == 0:
                logger.debug(
                    "Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: Loss: {:.4f} Accuracy: {:.4f}".format(
                        epoch,
                        phase,
                        idx,
                        len(loader),
                        running_loss / running_n,
                        running_corrects.double() / running_n,
                    )
                )

        epoch_loss = running_loss / running_n
        epoch_accuracy = running_corrects.double() / running_n

        if not (phase == "train"):
            prediction_file.write(
                "Epoch {:03d} Accuracy: {:.4f}\n".format(epoch, epoch_accuracy)
            )
            prediction_file.close()

        logger.info(
            "Epoch {:03d} Phase: {:10s} Loss: {:.4f} Accuracy: {:.4f}".format(
                epoch, phase, epoch_loss, epoch_accuracy
            )
        )

        return epoch_loss, epoch_accuracy

    def train(self, n_epochs, datasets, device, model_path, predictions_path):
        """
        Train the model
        Args:
            n_epochs (int): number of epochs to be run
            datasets: DataSet object with datasets loaded
            device (str): which device it is being run on. 'cpu' or 'cuda'
            model_path (str): path the save model weights to
            predictions_path (str): path to write predictions to
        """
        train_set, train_loader = datasets.load("train")
        validation_set, validation_loader = datasets.load("validation")

        best_accuracy, best_model_wts = 0.0, copy.deepcopy(self.model.state_dict())

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            # train network
            train_loss, train_accuracy = self.run_epoch(
                epoch, train_loader, device, predictions_path, phase="train"
            )

            # eval on validation
            validation_loss, validation_accuracy = self.run_epoch(
                epoch, validation_loader, device, predictions_path, phase="validation"
            )

            self.writer.add_scalar('Train/Loss', train_loss, epoch)
            self.writer.add_scalar('Train/Accuracy', train_accuracy, epoch)
            self.writer.add_scalar('Validation/Loss', validation_loss, epoch)
            self.writer.add_scalar('Validation/Accuracy', validation_accuracy, epoch)
            #Used to make sure tensorboard saves output at this point of time.
            #Somehow doesn't work gives "SummaryWriter" has no attribute "flush"
            # self.writer.flush()

            self.lr_scheduler.step(validation_loss)

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())

                logger.info(
                    "Epoch {:03d} Checkpoint: Saving to {}".format(epoch, model_path)
                )
                torch.save(best_model_wts, model_path)

        time_elapsed = time.time() - start_time
        logger.info(
            "Training complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

        logger.info("Best validation Accuracy: {:4f}.".format(best_accuracy))

    def test(self, datasets, device, model_path, predictions_path):
        """
        Test the model
        Args:
            datasets: DataSet object with datasets loaded
            device (str): which device it is being run on. 'cpu' or 'cuda'
            model_path (str): path to retrieve the saved model weights from
            predictions_path (str): path to write predictions to
        """
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        test_set, test_loader = datasets.load("test")
        test_loss, test_accuracy=self.run_epoch(1, test_loader, device, predictions_path, phase="test")
        self.writer.add_scalar('Test/Loss', test_loss, epoch)
        self.writer.add_scalar('Test/Accuracy', test_accuracy, epoch)