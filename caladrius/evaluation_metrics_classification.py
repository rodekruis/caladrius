import pandas as pd
import numpy as np
import re
import argparse
import os
from os import fdopen, remove
from tempfile import mkstemp
from shutil import move
import pickle

import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    recall_score,
    classification_report,
    roc_curve,
    auc,
    accuracy_score,
)
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix


def create_confusionmatrix(
    y_true, y_pred, filename, labels, figsize=(10, 10), class_names=None
):
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
    if class_names is None:
        class_names = labels

    fig, ax = plot_confusion_matrix(
        conf_mat=cm,
        colorbar=True,
        show_absolute=True,  # False,
        show_normed=True,
        class_names=class_names,
    )
    ax.margins(2, 2)
    plt.tight_layout()
    # plt.show()
    # fig.savefig("../../DataAnalysis/Data/conf_matrix_7disasters.pdf")
    # print(filename)

    fig.savefig(filename, bbox_inches="tight")


def harmonic_score(scores):
    """
    Calculate the harmonic mean of a list of scores
    Args:
        scores (list): list of scores

    Returns:
        harmonic mean of scores
    """
    return len(scores) / sum((c + 1e-6) ** -1 for c in scores)


def gen_score_overview(preds_filename, binary=False):
    """
    Generate a dataframe with several performance measures
    Args:
        preds_filename: name of file where predictions are saved

    Returns:
        score_overview (pd.DataFrame): dataframe with several performance measures
        df_pred (pd.DataFrame): dataframe with the predictions and true labels
    """

    if not binary:
        damage_mapping = {
            "0": "No damage",
            "1": "Minor damage",
            "2": "Major damage",
            "3": "Destroyed",
        }

    else:
        damage_mapping = {
            "0": "No damage",
            "1": "Damage",
        }

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
    unique_labels = np.unique(labels)
    damage_labels = [i for i in unique_labels if i != 0]
    # print(sorted(df_pred.label.unique())>0)
    print(unique_labels)
    print(damage_mapping.keys())
    print(list(map(int, damage_mapping.keys())))
    report = classification_report(
        labels,
        preds,
        digits=3,
        output_dict=True,
        labels=list(map(int, damage_mapping.keys())),
    )
    score_overview = pd.DataFrame(report).transpose()
    # print(score_overview)
    score_overview = score_overview.append(pd.Series(name="harmonized avg"))

    score_overview.loc["harmonized avg", ["precision", "recall", "f1-score"]] = [
        harmonic_score(r)
        for i, r in score_overview.loc[
            list(map(str, unique_labels)), ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    # create report only for damage categories (represented by 1,2,3)
    dam_report = classification_report(
        labels, preds, labels=damage_labels, output_dict=True
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
            list(map(str, damage_labels)), ["precision", "recall", "f1-score"]
        ].T.iterrows()
    ]

    if damage_mapping:
        score_overview.rename(index=damage_mapping, inplace=True)
    return score_overview, df_pred, damage_mapping


def create_overviewdict(df_overview, damage_mapping):
    # scores_params = [
    #     "harmonized_f1",
    #     "macro f1"
    #     "macro recall",
    #     # "harmonized_recall_damage",
    #     # "weighted_recall_damage",
    #     # "macro_recall_damage",
    #     # "support damage",
    #     "number datapoints",
    #     "percentages classes",
    #     # "percentage damage",
    # ]

    perc_dam = {}
    scores_dict = {}  # dict.fromkeys(scores_params)

    # save overview params
    # scores_dict["accuracy"] = df_overview.loc["accuracy","precision"]
    scores_dict["macro_f1"] = df_overview.loc["macro avg", "f1-score"]
    scores_dict["harmonized_f1"] = df_overview.loc["harmonized avg", "f1-score"]

    scores_dict["macro recall"] = df_overview.loc["macro avg", "recall"]
    scores_dict["macro precision"] = df_overview.loc["macro avg", "precision"]
    # scores_dict["harmonized_recall_damage"] = df_overview.loc[
    #     "damage harmonized avg", "recall"
    # ]
    # scores_dict["weighted_recall_damage"] = df_overview.loc[
    #     "damage weighted avg", "recall"
    # ]
    # scores_dict["macro_recall_damage"] = df_overview.loc["damage macro avg", "recall"]
    # scores_dict["support damage"] = int(df_overview.loc["damage macro avg", "support"])

    scores_dict = {
        k: round(v, 3) if v is not None else "" for k, v in scores_dict.items()
    }
    print(df_overview)
    for d in damage_mapping.values():
        scores_dict["recall {}".format(d)] = round(df_overview.loc[d, "recall"], 3)
        perc_dam[d] = round(
            df_overview.loc[d, "support"]
            / df_overview.loc["macro avg", "support"]
            * 100,
            1,
        )

    scores_dict["class percentage"] = "/".join(map(str, perc_dam.values()))
    scores_dict["number datapoints"] = int(df_overview.loc["macro avg", "support"])
    # scores_dict["percentage damage"] = round(
    #     scores_dict["support damage"] / scores_dict["support all"] * 100, 1
    # )
    return scores_dict


def plot_distrs(outputs, df_pred):
    # plot probability distribution for binary labels
    fig = plt.figure(figsize=(12, 9), constrained_layout=True)
    sns.distplot(
        outputs[df_pred.index[(np.array(df_pred.label) == 0)]][:, 1],
        label="No damage",
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        bins=int(180 / 5),
        color="darkgreen",
    )
    sns.distplot(
        outputs[df_pred.index[(np.array(df_pred.label) == 1)]][:, 1],
        label="Damage",
        hist=False,
        kde=True,
        kde_kws={"shade": True, "linewidth": 3},
        bins=int(180 / 5),
        color="red",
    )
    return fig


def calc_prob(preds_filename_prob, df_pred, binary=False):

    preds_file_probability = open(preds_filename_prob, "rb")
    outputs = pickle.load(preds_file_probability)
    outputs = np.array(outputs)
    print("shape outputs all", outputs.shape)
    preds_file_probability.close()

    preds = np.array(df_pred.pred)
    labels = np.array(df_pred.label)
    df_bin = df_pred.copy()
    if not binary:

        df_bin.label = df_bin.label.replace([2, 3], 1)
        df_bin.pred = df_bin.pred.replace([2, 3], 1)
        labels_bin = np.array(df_bin.label)
        preds_bin = np.array(df_bin.pred)
        outputs_bin = np.empty([len(outputs), 2])
        outputs_bin[:, 0] = outputs[:, 0]
        outputs_bin[:, 1] = outputs[:, 1:].sum(axis=1)

    else:
        labels_bin = labels
        outputs_bin = outputs
        preds_bin = preds

    print("shape outputs", outputs_bin.shape)
    print("shape labels", labels_bin.shape)
    # accuracy = accuracy_score(labels_bin, preds_bin, normalize=True)

    fpr, tpr, thresholds = roc_curve(labels_bin, outputs_bin[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, axes = plt.subplots(1, 1, figsize=(12, 9), constrained_layout=True)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.legend(loc="lower right")
    plt.setp(
        axes,
        xlim=[0.0, 1.0],
        ylim=[0.0, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
    )

    # ax.margins(2, 2)
    # plt.tight_layout()
    # plt.show()

    report = classification_report(labels_bin, preds_bin, digits=3, output_dict=True)
    scores_dict = {}
    scores_dict["accuracy"] = round(report["accuracy"], 3)
    scores_dict["auc"] = round(roc_auc, 3)
    scores_dict["recall_damage"] = round(report["1"]["recall"], 3)
    scores_dict["macro_precision"] = round(report["macro avg"]["precision"], 3)
    scores_dict["macro_recall"] = round(report["macro avg"]["recall"], 3)
    scores_dict["macro_f1"] = round(report["macro avg"]["f1-score"], 3)

    print(scores_dict)
    # print(report)
    fig_distr = plot_distrs(outputs_bin, df_bin)
    # plt.show()
    # fig.savefig("../../DataAnalysis/Data/conf_matrix_7disasters.pdf")
    # print(filename)

    return df_bin, scores_dict, fig_roc, fig_distr


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
    # scores_dict_rounded = {
    #     k: round(v, 3) if v is not None else "" for k, v in overview_dict.items()
    # }
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
                                    str(item) for item in list(overview_dict.values())
                                ),
                            )
                        )
                    else:
                        new_file.write(line)
        if not replicate:
            new_file.write(
                "{},{}\n".format(
                    run_name,
                    ",".join(str(item) for item in list(overview_dict.values())),
                )
            )
    if os.path.isfile(overview_path):
        remove(overview_path)
    move(abs_path, overview_path)


def main():
    NEURAL_MODELS = ["siamese", "inception", "light", "probability", "after", "shared"]
    STATISTICAL_MODELS = ["average", "random"]

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

    parser.add_argument(
        "--binary", default=False, action="store_true", help="If input data is binary",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default=NEURAL_MODELS[0],
        choices=NEURAL_MODELS + STATISTICAL_MODELS,
        help="type of model",
    )

    args = parser.parse_args()
    if not args.run_folder:
        args.run_folder = os.path.join(
            args.checkpoint_folder,
            "{}-input_size_32-learning_rate_0.001-batch_size_32".format(args.run_name),
        )

    # define all file names and paths
    test_file_name = "{}-split_test-epoch_001-model_{}-predictions.txt".format(
        args.run_name, args.model_type
    )
    preds_model = "{}/predictions/{}".format(args.run_folder, test_file_name)
    preds_random = "{}/predictions/{}-split_test-epoch_001-model_random-predictions.txt".format(
        args.run_folder, args.run_name
    )
    preds_average = "{}/predictions/{}-split_test-epoch_001-model_average-predictions.txt".format(
        args.run_folder, args.run_name
    )
    preds_probability = "{}/predictions/{}-split_test-epoch_001-model_probability-predictions.txt".format(
        args.run_folder, args.run_name
    )
    preds_validation = "{}/predictions/{}-split_validation-epoch_100-model_{}-predictions.txt".format(
        args.run_folder, args.run_name, args.model_type
    )
    output_path = "./performance/"
    score_overviews_path = os.path.join(output_path, "score_overviews/")
    confusion_matrices_path = os.path.join(output_path, "confusion_matrices/")
    confusion_matrices_path_bin = os.path.join(
        output_path, "confusion_matrices_binary/"
    )
    roc_curves_path = os.path.join(output_path, "roc_curves/")
    distr_plots_path = os.path.join(output_path, "distribution_plots/")

    for p in [
        output_path,
        score_overviews_path,
        confusion_matrices_path,
        confusion_matrices_path_bin,
        roc_curves_path,
        distr_plots_path,
    ]:
        if not os.path.exists(p):
            os.makedirs(p)

    for preds_filename, preds_type in zip(
        [preds_model, preds_random, preds_average, preds_probability, preds_validation],
        ["model", "random", "average", "probability", "validation"],
    ):
        # print(preds_filename)
        # check if file for preds type exists
        if os.path.exists(preds_filename):
            # generate overview with performance measures
            if preds_type != "probability":
                score_overview, df_pred, damage_mapping = gen_score_overview(
                    preds_filename, args.binary
                )

                score_overview.to_csv(
                    "{}{}_overview_{}.csv".format(
                        score_overviews_path, args.run_name, preds_type
                    )
                )
            else:
                _, df_pred, _ = gen_score_overview(preds_model, args.binary)
                df_pred_bin, prob_dict, roc_fig, dist_fig = calc_prob(
                    preds_probability, df_pred, args.binary
                )
                unique_labels_bin = np.unique(np.array(df_pred_bin.label))
                save_overviewfile(
                    prob_dict,
                    args.run_name,
                    output_path,
                    filename="allruns_scores_prob.txt",
                )
                create_confusionmatrix(
                    df_pred_bin.label,
                    df_pred_bin.pred,
                    "{}{}_confusion".format(confusion_matrices_path_bin, args.run_name),
                    unique_labels_bin,
                    class_names=["No damage", "Damage"],
                    figsize=(9, 12),
                )
                roc_fig.savefig(
                    "{}{}_roccurve".format(roc_curves_path, args.run_name),
                    bbox_inches="tight",
                )
                dist_fig.savefig(
                    "{}{}_distribution".format(distr_plots_path, args.run_name),
                    bbox_inches="tight",
                )

            if preds_type == "model":
                scores_dict = create_overviewdict(score_overview, damage_mapping)
                if args.binary:
                    filename_allscores = "allruns_scores_binary.txt"
                else:
                    filename_allscores = "allruns_scores.txt"
                save_overviewfile(
                    scores_dict,
                    args.run_name,
                    output_path,
                    filename=filename_allscores,
                )
            if preds_type in ["model", "validation"]:
                print(preds_type)
                unique_labels = np.unique(np.array(df_pred.label))
                # generate and save confusion matrix
                create_confusionmatrix(
                    df_pred.label,
                    df_pred.pred,
                    "{}{}_confusion_{}".format(
                        confusion_matrices_path, args.run_name, preds_type
                    ),
                    unique_labels,
                    figsize=(9, 12),
                )

        else:
            print("No predictions for prediction type {}".format(preds_type))


if __name__ == "__main__":
    main()
