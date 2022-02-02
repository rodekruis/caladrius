import os
import sys
import argparse
import logging
import pandas as pd


def binary_labels(
    directory_path,
    file_label_in,
    file_label_out,
    switch=False,
    destroyed=False,
    destroyed_switch=False,
):
    for set_name in ["train", "validation", "test"]:
        df = pd.read_csv(
            os.path.join(directory_path, set_name, file_label_in),
            sep=" ",
            header=None,
            names=["filename", "damage"],
        )
        if switch:
            df.damage = (df.damage < 1).astype(int)
        elif destroyed:
            df.damage = (df.damage > 2).astype(int)
        elif destroyed_switch:
            df.damage = (df.damage < 3).astype(int)
        else:
            df.damage = (df.damage >= 1).astype(int)

        df.to_csv(
            os.path.join(directory_path, set_name, file_label_out),
            sep=" ",
            index=False,
            header=False,
        )


def disaster_labels(disaster_names, directory_path, file_label_in, file_label_out):
    assert disaster_names is not None

    for set_name in ["train", "validation", "test"]:
        label_path = os.path.join(directory_path, set_name, file_label_in)
        if os.path.exists(label_path):
            df = pd.read_csv(
                label_path, sep=" ", header=None, names=["filename", "damage"],
            )
            disaster_names_list = [item for item in disaster_names.split(",")]
            pattern = "|".join([f"{d}" for d in disaster_names_list])
            df_select = df[df.filename.str.contains(pattern)]
            df_select.to_csv(
                os.path.join(directory_path, set_name, file_label_out),
                sep=" ",
                index=False,
                header=False,
            )
        else:
            print("No label file for {}".format(set_name))


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
        choices=[
            "binary",
            "regression",
            "regression_noise",
            "disaster",
            "binary_switch",
            "binary_des",
            "binary_des_switch",
        ],
        help="type of output labels",
    )

    parser.add_argument(
        "--disaster-names",
        default=None,
        type=str,
        metavar="disaster_names",
        help="List of disasters to be included, as a delimited string. E.g. typhoon,flood This can be types or specific occurences, as long as the building filenames contain these names.",
    )

    args = parser.parse_args()

    if args.label_type == "binary":
        binary_labels(args.data_path, args.file_in, args.file_out)
    elif args.label_type == "binary_switch":
        binary_labels(args.data_path, args.file_in, args.file_out, switch=True)
    elif args.label_type == "binary_des":
        binary_labels(args.data_path, args.file_in, args.file_out, destroyed=True)
    elif args.label_type == "binary_des_switch":
        binary_labels(
            args.data_path, args.file_in, args.file_out, destroyed_switch=True
        )

    elif args.label_type == "disaster":
        disaster_labels(
            args.disaster_names, args.data_path, args.file_in, args.file_out,
        )


if __name__ == "__main__":
    main()
