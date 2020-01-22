from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import torch

from utils import create_logger


logger = create_logger(__name__)


class RollingEval(object):
    def __init__(self, output_type):
        self.output_type = output_type
        self.labels = torch.Tensor([])
        self.predictions = torch.Tensor([])
        self.total_loss = 0.0

        # default damage class boundaries
        self.upper_bound = 0.7
        self.lower_bound = 0.3

    def add(self, labels, predictions, loss):
        self.labels = self.labels.to(labels.device)
        self.predictions = self.predictions.to(predictions.device)
        if self.output_type == "regression":
            self.labels = self.labels.float()
            self.predictions = self.predictions.float()
            labels = labels.float()
            predictions = predictions.float()
        else:
            self.labels = self.labels.long()
            self.predictions = self.predictions.long()
            labels = labels.long()
            predictions = predictions.long()
        self.labels = torch.cat([self.labels, labels], dim=0)
        self.predictions = torch.cat([self.predictions, predictions], dim=0)
        self.total_loss += loss * predictions.size(0)
        batch_score = self.score((labels, predictions))
        return batch_score

    def to_classes(self, score):
        score[score >= self.upper_bound] = 2
        score[(score > self.lower_bound) & (score < self.upper_bound)] = 1
        score[score <= self.lower_bound] = 0
        return score

    def score(self, labels_predictions=None):
        if labels_predictions is None:
            labels_predictions = (self.labels, self.predictions)
        labels, predictions = labels_predictions
        if self.output_type == "regression":
            labels = self.to_classes(labels)
            predictions = self.to_classes(predictions)
        return self.precision_recall_fscore_support_accuracy(labels, predictions)

    def loss(self):
        return self.total_loss / self.predictions.size(0)

    def precision_recall_fscore_support_accuracy(self, labels, predictions):
        # average="weighted" is used by xview2_baseline
        # average="micro" was used by caladrius
        labels = labels.cpu().detach()
        predictions = predictions.cpu().detach()
        accuracy_value = accuracy_score(labels, predictions, normalize=True)
        number_of_correct = accuracy_score(labels, predictions, normalize=False)
        total_number = len(predictions)
        micro_precision_recall_fscore_support_values = precision_recall_fscore_support(
            labels, predictions, average="micro"
        )
        macro_precision_recall_fscore_support_values = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        weighted_precision_recall_fscore_support_values = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        return (
            accuracy_value,
            number_of_correct,
            total_number,
            micro_precision_recall_fscore_support_values,
            macro_precision_recall_fscore_support_values,
            weighted_precision_recall_fscore_support_values,
        )
