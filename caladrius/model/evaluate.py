from sklearn.metrics import f1_score
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
        return self.f1_score(labels, predictions)

    def loss(self):
        return self.total_loss / self.predictions.size(0)

    def f1_score(self, labels, predictions):
        # average="weighted" is used by xview2_baseline
        # average="micro" was used by caladrius
        return f1_score(
            labels.cpu().detach(), predictions.cpu().detach(), average="macro"
        )
