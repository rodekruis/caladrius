import shutil
import os
import matplotlib.pyplot as plt
from skimage import exposure
import argparse


def loop_hist(wd):
    src_dir = "{}_histequal".format(wd)
    if not os.path.exists(src_dir):
        shutil.copytree(wd, src_dir)
    img_files = []
    for root, directories, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename.endswith((".png")):
                img_files.append(os.path.join(root, filename))

    for i in img_files:
        img = plt.imread(i)
        img_adj = exposure.equalize_adapthist(img, clip_limit=0.05)
        plt.imsave(fname=i, arr=img_adj, cmap="gray")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path where the images that need to be equalized are stored, can contain subfolders",
    )

    args = parser.parse_args()

    loop_hist(args.data_path)

    # loop_hist("../data/minitest_out_class")
    # loop_hist("../data/minitest_pre")
