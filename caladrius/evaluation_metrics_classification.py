import pandas as pd
import numpy as np
# from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import argparse

def tp_fn_fp_perclass(preds, labels, c):
    """
    Computes the number of TPs, FNs, FPs, between a prediction (x) and a target (y) for the desired class (c)
    Args:
        pred (np.ndarray): prediction
        targ (np.ndarray): target
        c (int): positive class
    """
    TP = np.logical_and(preds == c, labels == c).sum()
    FN = np.logical_and(preds != c, labels == c).sum()
    FP = np.logical_and(preds == c, labels != c).sum()
    return [TP, FN, FP]

def precision(TP,FP):
    assert TP >= 0 and FP >= 0
#     if TP ==0 and FP ==0:
#         return 1
    if TP == 0:
        return 0
    else:
        return TP/(TP+FP)
def recall(TP,FN):
    assert TP>=0 and FN >=0
#     if TP==0 and FN==0:
#         return 1
    if TP==0:
        return 0
    else:
        return TP/(TP+FN)
def f1_score(PR,RE):
    return 2*((PR*RE)/(PR+RE+1e-6))

def harmonic_score(scores):
#     print(sum((c+1e-6)**-1 for c in scores))
    return len(scores)/sum((c+1e-6)**-1 for c in scores)

def harmonic_scores2(scores):
    print(scores)
    print(sum(c**-1 for c in scores))
    print(sum(c for c in scores))
    print(sum(c for c in scores)/len(scores))
    return len(scores)/(sum(c**-1 for c in scores))

def df1(self):
    """ damage f1. Computed using harmonic mean of damage f1s """
    harmonic_mean = lambda xs: len(xs) / sum((x+1e-6)**-1 for x in xs)
    return harmonic_mean(self.df1s)


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
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
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    # plt.show()
    plt.savefig(filename,bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="name to identify execution",
    )

    parser.add_argument(
        "--run-folder",
        # required=True,
        #default=os.path.join('../data', 'xBD'),
        metavar="Folder name of outputs of run",
        help="Full path to the directory with txt file with labels and predictions",
    )

    args = parser.parse_args()
    if not args.run_folder:
        args.run_folder="{}-input_size_32-learning_rate_0.001-batch_size_32".format(
        args.run_name
    )
    test_file_name="{}_test_epoch_001_predictions.txt".format(args.run_name)
    preds_filename="../runs/{}/predictions/{}".format(args.run_folder,test_file_name)
    output_path="../runs/{}/performance/".format(args.run_folder)

    preds_file=open(preds_filename)
    lines=preds_file.readlines()[1:-1]
    pred_info=[]
    for l in lines:
        pred_info.extend([l.rstrip().split(" ")])
    df_pred=pd.DataFrame(pred_info,columns=["OBJECTID","label","pred"])
    df_pred.label=df_pred.label.astype(int)
    df_pred.pred=df_pred.pred.astype(int)

    damage_mapping = {0: "No damage", 1: "Minor damage", 2: "Major damage", 3: "Destroyed"}

    preds = np.array(df_pred.pred)
    labels = np.array(df_pred.label)
    # scores_classes = {"recall": {}, "precision": {}, "f1": {}}
    # for c in damage_mapping.keys():
    #     tp, fn, fp = tp_fn_fp_perclass(preds, labels, c)
    #     scores_classes["recall"][c] = recall(tp, fn)
    #     scores_classes["precision"][c] = precision(tp, fp)
    #     scores_classes["f1"][c] = f1_score(scores_classes["precision"][c], scores_classes["recall"][c])
    #
    # score_overview = pd.DataFrame(scores_classes).T
    # score_overview.rename(columns=damage_mapping, inplace=True)
    # score_overview["Total"] = [harmonic_score(r) for i, r in score_overview.iterrows()]

    report = classification_report(preds, labels, digits=3,output_dict=True)
    score_overview = pd.DataFrame(report).transpose()
    print(score_overview.index)
    damage_mapping = {'0': "No damage", '1': "Minor damage", '2': "Major damage", '3': "Destroyed"}
    score_overview.rename(index=damage_mapping,inplace=True)


    score_overview.to_csv("{}score_overview.csv".format(output_path))

    # energy.rename(index={'Republic of Korea': 'South Korea'}, inplace=True)

    cm_analysis(df_pred.label, df_pred.pred, "{}confusion_matrix".format(output_path), [0, 1, 2, 3],figsize=(9,12))



    # print(score_overview)
    # print(harmonic_scores2(np.array(score_overview.loc["f1",["No damage","Minor damage","Major damage","Destroyed"]])))
    # print(harmonic_scores2(np.array([0.5,0.25])))
if __name__ == '__main__':
    main()