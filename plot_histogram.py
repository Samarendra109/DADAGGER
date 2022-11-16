import numpy as np
import matplotlib.pyplot as plt
import re
from os import listdir
import argparse


def plot_histogram(args):
    filenames = [f for f in listdir(args.folder) if f.endswith("jpg")]
    re_pattern = "expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg"
    steering_command = np.array(
        [float(re.search(re_pattern, basename).group(3)) for basename in filenames]
    )

    counts, bins = np.histogram(steering_command, bins=args.n_steering_classes)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.hist(bins[:-1], bins, weights=counts)
    ax.set_xlabel("Expert steering command")
    fig.savefig(f"{args.folder}/histogram_of_samples.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_steering_classes", type=int, help="number of steering classes", default=20
    )
    parser.add_argument(
        "--folder",
        help="directory of training data produced after a dagger run",
        default="./dataset/train",
    )

    args = parser.parse_args()

    plot_histogram(args)
