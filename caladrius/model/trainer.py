import os
import copy
import time
import torch

from torch.optim import Adam
from torch.nn.modules import loss as nnloss
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from caladrius.model.network import get_pretrained_iv3_transforms, SiameseNetwork
from caladrius.utils import create_logger


logger = create_logger(__name__)


class QuasiSiameseNetwork(object):

    def __init__(self, args):
        input_size = (args.inputSize, args.inputSize)

        self.run_name = args.runName
        self.input_size = input_size
        self.lr = args.learningRate

        self.criterion = nnloss.MSELoss()

        self.transforms = {}

        self.model = SiameseNetwork()

        if torch.cuda.device_count() > 1:
            logger.info('Using {} GPUs'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)

        for s in ('train', 'validation', 'test'):
            self.transforms[s] = get_pretrained_iv3_transforms(s)

        logger.debug('Num params: {}'.format(
            len([_ for _ in self.model.parameters()])))

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer,
                                              factor=0.1,
                                              patience=10,
                                              min_lr=1e-5,
                                              verbose=True)

    def run_epoch(self, epoch, loader, device, phase='train', accuracy_threshold=0.1):
        assert phase in ('train', 'validation', 'test')

        self.model = self.model.to(device)

        self.model.eval()
        if phase == 'train':
            self.model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects = 0
        running_n = 0.0

        if not (phase == 'train'):
            prediction_file_name = '{}_epoch_{:03d}_predictions.txt'.format(self.run_name, epoch)
            prediction_file_path = os.path.join(loader.dataset.directory, prediction_file_name)
            prediction_file = open(prediction_file_path, 'w+')
            prediction_file.write('filename label prediction\n')

        for idx, (filename, image1, image2, labels) in enumerate(loader, 1):
            image1 = image1.to(device)
            image2 = image2.to(device)
            labels = labels.float().to(device)

            if phase == 'train':
                # zero the parameter gradients
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(image1, image2).squeeze()
                loss = self.criterion(outputs, labels)

                if not (phase == 'train'):
                    prediction_file.writelines([ '{} {} {}\n'.format(*line) for line in zip(filename, labels.view(-1).tolist(), outputs.clamp(0, 1).view(-1).tolist())])

                if phase == 'train':
                    loss.backward()
                    self.optimizer.step()

            running_loss += loss.item() * image1.size(0)
            running_corrects += (outputs - labels.data).abs().le(accuracy_threshold).sum()
            running_n += image1.size(0)

            if idx % 1 == 0:
                logger.debug('Epoch: {:03d} Phase: {:10s} Batch {:04d}/{:04d}: Loss: {:.4f} Accuracy: {:.4f}'.format(
                    epoch, phase, idx, len(loader), running_loss / running_n, running_corrects.double() / running_n))

        epoch_loss = running_loss / running_n
        epoch_accuracy = running_corrects.double() / running_n

        if not (phase == 'train'):
            prediction_file.write('Epoch {:03d} Accuracy: {:.4f}\n'.format(epoch, epoch_accuracy))
            prediction_file.close()

        logger.info('Epoch {:03d} Phase: {:10s} Loss: {:.4f} Accuracy: {:.4f}'.format(epoch, phase, epoch_loss, epoch_accuracy))

        return epoch_loss, epoch_accuracy

    def train(self, n_epochs, datasets, device, save_path):
        train_set, train_loader = datasets.load('train')
        validation_set, validation_loader = datasets.load('validation')

        best_accuracy, best_model_wts = 0.0, copy.deepcopy(
            self.model.state_dict())

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            # train network
            train_loss, train_accuracy = self.run_epoch(
                epoch, train_loader, device, phase='train')

            # eval on validation
            validation_loss, validation_accuracy = self.run_epoch(
                epoch, validation_loader, device, phase='validation')

            self.lr_scheduler.step(validation_loss)

            if validation_accuracy > best_accuracy:
                best_accuracy = validation_accuracy
                best_model_wts = copy.deepcopy(self.model.state_dict())

                logger.info('Epoch {:03d} Checkpoint: Saving to {}'.format(epoch, save_path))
                torch.save(best_model_wts, save_path)

        time_elapsed = time.time() - start_time
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        logger.info('Best validation Accuracy: {:4f}.'.format(best_accuracy))

    def test(self, datasets, device, load_path):
        self.model.load_state_dict(torch.load(load_path, map_location=device))
        test_set, test_loader = datasets.load('test')
        self.run_epoch(1, test_loader, device, phase='test')
