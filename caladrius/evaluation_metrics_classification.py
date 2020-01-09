import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import argparse
import os

from tempfile import mkstemp
from shutil import move
from os import fdopen, remove


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    Copied from https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt="", ax=ax)
    plt.savefig(filename, bbox_inches="tight")


def harmonic_score(scores):
    """
    Calculate the harmonic mean of a list of scores
    Args:
        scores (list): list of scores

    Returns:
        harmonic mean of scores
    """
    return len(scores) / sum((c + 1e-6) ** -1 for c in scores)


def different_averages(scores):
    harmonic_avg = len(scores) / sum((c + 1e-6) ** -1 for c in scores)
    macro_avg = sum(scores) / len(scores)
    return harmonic_avg, macro_avg


def gen_score_overview(preds_filename):
    """
    Generate a dataframe with several performance measures
    Args:
        preds_filename: name of file where predictions are saved

    Returns:
        score_overview (pd.DataFrame): dataframe with several performance measures
        df_pred (pd.DataFrame): dataframe with the predictions and true labels
    """
    preds_file = open(preds_filename)
    lines = preds_file.readlines()[1:]
    pred_info = []
    for l in lines:
        pred_info.extend([l.rstrip().split(" ")])
    df_pred = pd.DataFrame(pred_info, columns=["OBJECTID", "label", "pred"])
    df_pred.label = df_pred.label.astype(int)
    df_pred.pred = df_pred.pred.astype(int)

    preds = np.array(df_pred.pred)
    labels = np.array(df_pred.label)

    report = classification_report(labels, preds, digits=3, output_dict=True)
    # print(report)
    dam_report = classification_report(
        labels, preds, labels=[1, 2, 3], output_dict=True
    )
    # print(dam_report.keys())
    dam_report = pd.DataFrame(dam_report).transpose()

    score_overview = pd.DataFrame(report).transpose()

    score_overview = score_overview.append(pd.Series(name="harmonized avg"))
    score_overview = score_overview.append(pd.Series(name="damage macro avg"))
    score_overview = score_overview.append(pd.Series(name="damage weighted avg"))
    score_overview = score_overview.append(pd.Series(name="damage harmonized avg"))
    score_overview.loc["harmonized avg", ["precision", "recall", "f1-score"]] = [
        harmonic_score(r)
        for i, r in score_overview.loc[
            ["0", "1", "2", "3"], ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    score_overview.loc[
        "damage macro avg", ["precision", "recall", "f1-score", "support"]
    ] = (
        dam_report.loc[["macro avg"], ["precision", "recall", "f1-score", "support"]]
        .values.flatten()
        .tolist()
    )

    score_overview.loc[
        "damage weighted avg", ["precision", "recall", "f1-score", "support"]
    ] = (
        dam_report.loc[["weighted avg"], ["precision", "recall", "f1-score", "support"]]
        .values.flatten()
        .tolist()
    )

    score_overview.loc["damage harmonized avg", ["precision", "recall", "f1-score"]] = [
        harmonic_score(r)
        for i, r in score_overview.loc[
            ["1", "2", "3"], ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    damage_mapping = {
        "0": "No damage",
        "1": "Minor damage",
        "2": "Major damage",
        "3": "Destroyed",
    }
    score_overview.rename(index=damage_mapping, inplace=True)
    return score_overview, df_pred


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-name", type=str, required=True, help="name to identify execution",
    )

    parser.add_argument(
        "--run-folder",
        metavar="Folder name of outputs of run",
        help="Full path to the directory with txt file with labels and predictions",
    )

    args = parser.parse_args()
    if not args.run_folder:
        args.run_folder = "{}-input_size_32-learning_rate_0.001-batch_size_32".format(
            args.run_name
        )

    # define all file names and paths
    test_file_name = "{}_test_epoch_001_predictions.txt".format(args.run_name)
    preds_model = "../runs/{}/predictions/{}".format(args.run_folder, test_file_name)
    preds_random = "{}_random.txt".format(preds_model[:-4])
    preds_average = "{}_average.txt".format(preds_model[:-4])
    output_path = "../performance/"
    score_overviews_path = os.path.join(output_path, "score_overviews/")
    confusion_matrices_path = os.path.join(output_path, "confusion_matrices/")
    for p in [output_path, score_overviews_path, confusion_matrices_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    # parameters that should be written to overview file
    scores_params = [
        "unweighted_recall",
        "weighted_recall",
        "harmonized_recall",
        "unweightedrecall_random",
        "unweightedrecall_average",
    ]
    scores_dict = dict.fromkeys(scores_params)

    for preds_filename, preds_type in zip(
        [preds_model, preds_random, preds_average], ["model", "random", "average"]
    ):
        # check if file for preds type exists
        if os.path.exists(preds_filename):
            # generate overview with performance measures
            score_overview, df_pred = gen_score_overview(preds_filename)
            score_overview.to_csv(
                "{}{}_overview_{}.csv".format(
                    score_overviews_path, args.run_name, preds_type
                )
            )
            print(
                "macro recall {}".format(preds_type),
                score_overview.loc["macro avg", "recall"],
            )
            print(
                "weighted recall {}".format(preds_type),
                score_overview.loc["weighted avg", "recall"],
            )

            if preds_type == "model":
                # generate and save confusion matrix
                cm_analysis(
                    df_pred.label,
                    df_pred.pred,
                    "{}{}_confusion".format(confusion_matrices_path, args.run_name),
                    [0, 1, 2, 3],
                    figsize=(9, 12),
                )
                # save overview params
                scores_dict["unweighted_recall"] = score_overview.loc[
                    "macro avg", "recall"
                ]
                scores_dict["weighted_recall"] = score_overview.loc[
                    "weighted avg", "recall"
                ]
                scores_dict["harmonized_recall"] = score_overview.loc[
                    "harmonized avg", "recall"
                ]
            if preds_type == "random":
                scores_dict["unweightedrecall_random"] = score_overview.loc[
                    "macro avg", "recall"
                ]
            if preds_type == "average":
                scores_dict["unweightedrecall_average"] = score_overview.loc[
                    "macro avg", "recall"
                ]
        else:
            print("No predictions for prediction type {}".format(preds_type))

    # save parameters to overview file
    # replace old values if line with same run_name already exists
    allruns_overview_file_name = "allruns_scores.txt"
    allruns_file_path = os.path.join(output_path, allruns_overview_file_name)
    fh, abs_path = mkstemp()
    replicate = False
    scores_dict_rounded = {
        k: round(v, 3) if v is not None else "" for k, v in scores_dict.items()
    }
    with fdopen(fh, "w+") as new_file:
        new_file.write(
            "run_name,{}\n".format(
                ",".join(str(item) for item in list(scores_dict.keys()))
            )
        )
        if os.path.isfile(allruns_file_path):
            with open(allruns_file_path) as old_file:
                next(old_file)
                for line in old_file:
                    if re.search(r"^{},".format(args.run_name), line):
                        replicate = True
                        new_file.write(
                            "{},{}\n".format(
                                args.run_name,
                                ",".join(
                                    str(item)
                                    for item in list(scores_dict_rounded.values())
                                ),
                            )
                        )
                    else:
                        new_file.write(line)
        if not replicate:
            new_file.write(
                "{},{}\n".format(
                    args.run_name,
                    ",".join(str(item) for item in list(scores_dict_rounded.values())),
                )
            )
    if os.path.isfile(allruns_file_path):
        remove(allruns_file_path)
    move(abs_path, allruns_file_path)


if __name__ == "__main__":
    main()
