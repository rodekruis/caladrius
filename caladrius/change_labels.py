import os
import sys
import argparse
import logging
import pandas as pd


def set_labels(directory_path, file_label_in, file_label_out):
    for set_name in ["train", "validation", "test"]:
        df = pd.read_csv(
            os.path.join(directory_path, set_name, file_label_in),
            sep=" ",
            header=None,
            names=["filename", "damage"],
        )
        df.damage = (df.damage >= 1).astype(int)
        df.to_csv(
            os.path.join(directory_path, set_name, file_label_out),
            sep=" ",
            index=False,
            header=False,
        )


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data-path",
        default=False,
        type=str,
        metavar="data_path",
        help="Path where buildings are saved",
    )
    parser.add_argument(
        "--file-in",
        default="labels.txt",
        type=str,
        metavar="file_in",
        help="name of file with original labels",
    )

    parser.add_argument(
        "--file-out",
        type=str,
        metavar="file_out",
        help="name of file with output labels",
    )

    parser.add_argument(
        "--label-type",
        default="binary",
        type=str,
        metavar="label_type",
        choices=["binary", "regression", "regression_noise"],
        help="type of output labels",
    )

    # parser.add_argument(
    #     "--label-values",
    #     default=["0","1","2","3"],
    #     metavar="label_values",
    #     help="unique values in input labels"
    # )

    args = parser.parse_args()

    set_labels(args.data_path, args.file_in, args.file_out, args.label_values)


if __name__ == "__main__":
    main()
