import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

D_csv = "./log/DCGAN_advanced/D_total.csv"
G_csv = "./log/DCGAN_advanced/G_total.csv"

D_data = pd.read_csv(D_csv)
G_data = pd.read_csv(G_csv)

plt.figure(figsize=(10, 6))

plt.plot(D_data["Step"], D_data["Value"], label="D_loss")
plt.plot(G_data["Step"], G_data["Value"], label="G_loss")
plt.title("Training Losses of DCGAN with Advanced Data Augmentation")
plt.xlabel("Step")
plt.ylabel("Training Losses")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)

plt.savefig("./log/DCGAN_advanced/loss.png")
