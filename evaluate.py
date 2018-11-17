import time
import json
import logging

import torch
from sklearn.metrics import f1_score

log = logging.getLogger(__name__)


class RollingEval(object):
    def __init__(self):
        self.y_true = []
        self.y_pred = []

    def add(self, y_t, y_p):
        print(y_t, y_p)
        self.y_true.extend(y_t.detach().numpy())
        self.y_pred.extend(y_p.detach().numpy())

    def f1_score(self):
        return f1_score(y_true, y_pred, average="micro")


class Evaluator(object):
    def __init__(self, model, sets):

        self.model = model
        # sets is a dictionary (set_name -> dataloader)
        self.datasets = sets

    def evaluate_set(self, set_name):
        assert set_name in self.datasets.keys()
        log.info("Starting evaluation of model")
        y_true, y_pred = self.gather_outputs(
            self.model, self.datasets[set_name])
        f1_score = f1_score(y_true, y_pred, average="micro")

        log.info("F1-Score ({}) : {}".format(set_name, f1_score))
        return f1_score

    def gather_outputs(self, model, dataset):
        y_true = []
        y_pred = []
        log.info(
            "Gathering inputs. Total number of datapoints: {}".format(len(dataset)))
        with torch.no_grad():
            for idx, (image1, image2, y_true_batch) in enumerate(dataset):
                y_pred_batch = model(image1, image2)

                y_pred_batch = y_pred_batch.argmax(1)
                y_pred_batch.extend(y_pred_batch.detach().numpy())
                y_true.extend(y_true_batch.detach().numpy())

        return y_true, y_pred

    def evaluate(self, results_path):
        results = {}
        for set_name, dataset in self.datasets.items():
            results[set_name] = self.evaluate_set(set_name)

        with open(results_path, "w") as writer:
            json.dump(writer, results, indent=2)
