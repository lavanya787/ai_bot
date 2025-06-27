import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("visualizer")

def moving_avg(data, window=3):
    return np.convolve(data, np.ones(window) / window, mode='valid')

def plot_loss(log_path="logs/training_log.txt", output_path="logs/training_loss_plot.png"):
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            logger.warning("No data to plot.")
            return

        epochs = df["Epoch"]
        train = df["TrainLoss"]
        val = df["ValLoss"]

        sm_train = moving_avg(train, 2)
        sm_val = moving_avg(val, 2)
        sm_epochs = epochs[-len(sm_train):]

        plt.figure(figsize=(8, 5))
        plt.plot(sm_epochs, sm_train, label="Train Loss", marker='o')
        plt.plot(sm_epochs, sm_val, label="Val Loss", marker='x')
        
        min_val_idx = np.argmin(sm_val)
        plt.annotate(f"Min Val: {sm_val[min_val_idx]:.4f}",
                     xy=(sm_epochs[min_val_idx], sm_val[min_val_idx]),
                     xytext=(sm_epochs[min_val_idx], sm_val[min_val_idx] + 0.1),
                     arrowprops=dict(arrowstyle="->"))

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        logger.info(f"Loss plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Plotting failed: {e}")
