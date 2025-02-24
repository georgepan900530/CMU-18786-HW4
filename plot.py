import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="./log/DCGAN_advanced")
    parser.add_argument("--file_name", type=str, default="loss.png")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.log_dir):
        print(f"Creating log directory {args.log_dir}")
        os.mkdir(args.log_dir)

    D_csv = os.path.join(args.log_dir, "D_total.csv")
    G_csv = os.path.join(args.log_dir, "G_total.csv")

    D_data = pd.read_csv(D_csv)
    G_data = pd.read_csv(G_csv)

    plt.figure(figsize=(10, 6))

    plt.plot(D_data["Step"], D_data["Value"], label="D_loss")
    plt.plot(G_data["Step"], G_data["Value"], label="G_loss")
    plt.title("Training Losses of DCGAN with Basic Data Augmentation")
    plt.xlabel("Step")
    plt.ylabel("Training Losses")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.savefig(os.path.join(args.log_dir, args.file_name))
