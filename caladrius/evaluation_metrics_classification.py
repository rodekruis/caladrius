import pandas as pd
import numpy as np
import re
import argparse
import os
from os import fdopen, remove
from tempfile import mkstemp
from shutil import move

import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def plot_confusionmatrix(y_true, y_pred, filename, labels, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    Adapted from https://gist.github.com/hitvoice/36cf44689065ca9b927431546381a3f7
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      figsize:   the size of the figure plotted.
    """

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
    score_overview = pd.DataFrame(report).transpose()

    score_overview = score_overview.append(pd.Series(name="harmonized avg"))
    score_overview.loc["harmonized avg", ["precision", "recall", "f1-score"]] = [
        harmonic_score(r)
        for i, r in score_overview.loc[
            ["0", "1", "2", "3"], ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    # create report only for damage categories (represented by 1,2,3)
    dam_report = classification_report(
        labels, preds, labels=[1, 2, 3], output_dict=True
    )
    dam_report = pd.DataFrame(dam_report).transpose()

    score_overview = score_overview.append(pd.Series(name="damage macro avg"))
    score_overview = score_overview.append(pd.Series(name="damage weighted avg"))
    score_overview = score_overview.append(pd.Series(name="damage harmonized avg"))

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


def create_overviewdict(df_overview):
    scores_params = [
        "harmonized_f1",
        "macro recall",
        "harmonized_recall_damage",
        "weighted_recall_damage",
        "macro_recall_damage",
        "support damage",
        "support all",
        "percentage damage",
    ]

    scores_dict = dict.fromkeys(scores_params)

    # save overview params
    scores_dict["harmonized_f1"] = df_overview.loc["harmonized avg", "f1-score"]
    scores_dict["macro recall"] = df_overview.loc["macro avg", "recall"]
    scores_dict["harmonized_recall_damage"] = df_overview.loc[
        "damage harmonized avg", "recall"
    ]
    scores_dict["weighted_recall_damage"] = df_overview.loc[
        "damage weighted avg", "recall"
    ]
    scores_dict["macro_recall_damage"] = df_overview.loc["damage macro avg", "recall"]
    scores_dict["support damage"] = int(df_overview.loc["damage macro avg", "support"])
    scores_dict["support all"] = int(df_overview.loc["macro avg", "support"])
    scores_dict["percentage damage"] = round(
        scores_dict["support damage"] / scores_dict["support all"] * 100, 1
    )
    return scores_dict


def save_overviewfile(
    overview_dict, run_name, output_path, filename="allruns_scores.txt"
):
    """
    Save overview_dict to a file with all other runs.
    Args:
        overview_dict: Keys and values that should be saved
        run_name: name of experiment
        output_path: path to output file
        filename: name of output file
    """
    # save parameters to overview file
    # replace old values if line with same run_name already exists
    overview_path = os.path.join(output_path, filename)
    fh, abs_path = mkstemp()
    replicate = False
    scores_dict_rounded = {
        k: round(v, 3) if v is not None else "" for k, v in overview_dict.items()
    }
    with fdopen(fh, "w+") as new_file:
        new_file.write(
            "run_name,{}\n".format(
                ",".join(str(item) for item in list(overview_dict.keys()))
            )
        )
        if os.path.isfile(overview_path):
            with open(overview_path) as old_file:
                next(old_file)
                for line in old_file:
                    if re.search(r"^{},".format(run_name), line):
                        replicate = True
                        new_file.write(
                            "{},{}\n".format(
                                run_name,
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
                    run_name,
                    ",".join(str(item) for item in list(scores_dict_rounded.values())),
                )
            )
    if os.path.isfile(overview_path):
        remove(overview_path)
    move(abs_path, overview_path)


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

    parser.add_argument(
        "--checkpoint-folder",
        type=str,
        default=os.path.join(".", "runs"),
        help="runs path",
    )

    args = parser.parse_args()
    if not args.run_folder:
        args.run_folder = os.path.join(
            args.checkpoint_folder,
            "{}-input_size_32-learning_rate_0.001-batch_size_32".format(args.run_name),
        )

    # define all file names and paths
    test_file_name = "{}-split_test-epoch_001-model_siamese-predictions.txt".format(
        args.run_name
    )
    preds_model = "{}/predictions/{}".format(args.run_folder, test_file_name)
    preds_random = "{}-split_test-epoch_001-model_random-predictions.txt".format(
        args.run_name
    )
    preds_average = "{}-split_test-epoch_001-model_average-predictions.txt".format(
        args.run_name
    )
    preds_validation = "{}-split_validation-epoch_100-model_siamese-predictions.txt".format(
        args.run_name
    )
    output_path = "./performance/"
    score_overviews_path = os.path.join(output_path, "score_overviews/")
    confusion_matrices_path = os.path.join(output_path, "confusion_matrices/")

    for p in [output_path, score_overviews_path, confusion_matrices_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    for preds_filename, preds_type in zip(
        [preds_model, preds_random, preds_average, preds_validation],
        ["model", "random", "average", "validation"],
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

            if preds_type == "model":
                scores_dict = create_overviewdict(score_overview)
                save_overviewfile(
                    scores_dict,
                    args.run_name,
                    output_path,
                    filename="allruns_scores.txt",
                )
                # generate and save confusion matrix
                plot_confusionmatrix(
                    df_pred.label,
                    df_pred.pred,
                    "{}{}_confusion".format(confusion_matrices_path, args.run_name),
                    [0, 1, 2, 3],
                    figsize=(9, 12),
                )

        else:
            print("No predictions for prediction type {}".format(preds_type))


if __name__ == "__main__":
    main()
