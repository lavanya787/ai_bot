import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Union
import json
import os
import seaborn as sns

try:
    from utils.domain_detector import detect_domain
    DOMAIN_DETECTION_AVAILABLE = True
except ImportError:
    DOMAIN_DETECTION_AVAILABLE = False
    detect_domain = lambda x: "general"

# Setup logging
logger = logging.getLogger("visualizer")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/visualizer.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

BASE_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
BASE_OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "rag_data"))
DEFAULT_WINDOW = 3
DEFAULT_PLOT_FORMAT = "png"
DEFAULT_DPI = 300
VALID_METRIC_SUFFIXES = ["Loss", "Accuracy", "F1", "Precision", "Recall"]
FALLBACK_STYLE = "default"

def get_valid_style(preferred_style: str = "seaborn-v0_8") -> str:
    available_styles = plt.style.available
    if preferred_style in available_styles:
        return preferred_style
    for style in ["seaborn-v0_12", "seaborn-v0_8", "ggplot", "default"]:
        if style in available_styles:
            logger.info(f"Using fallback style: {style}")
            return style
    logger.warning(f"No valid styles found. Using {FALLBACK_STYLE}")
    return FALLBACK_STYLE

def create_sample_log(log_path: Path) -> pd.DataFrame:
    sample_data = {
        "Epoch": [1, 2, 3],
        "TrainLoss": [0.693, 0.512, 0.401],
        "ValLoss": [0.682, 0.498, 0.390],
        "TrainAccuracy": [0.500, 0.620, 0.710],
        "ValAccuracy": [0.510, 0.630, 0.720]
    }
    df = pd.DataFrame(sample_data)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(log_path, index=False)
    logger.info(f"Created sample training log at {log_path}")
    return df

def moving_avg(data: np.ndarray, window: int = DEFAULT_WINDOW) -> np.ndarray:
    try:
        if len(data) < window:
            logger.warning(f"Data length ({len(data)}) is less than window size ({window}), returning raw data")
            return data
        return np.convolve(data, np.ones(window) / window, mode='valid')
    except Exception as e:
        logger.error(f"Error computing moving average: {e}")
        return data

def load_training_log(log_path: Union[str, Path]) -> pd.DataFrame:
    log_path = Path(log_path)
    try:
        if not log_path.exists():
            logger.warning(f"Training log file {log_path} does not exist. Creating sample log.")
            return create_sample_log(log_path)

        if log_path.suffix.lower() in [".csv", ".txt"]:
            df = pd.read_csv(log_path, encoding='utf-8')
        elif log_path.suffix.lower() == ".json":
            with open(log_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        else:
            logger.error(f"Unsupported file format: {log_path.suffix}")
            return pd.DataFrame()

        if df.empty:
            logger.warning(f"Training log {log_path} is empty. Creating sample log.")
            return create_sample_log(log_path)

        df = df.dropna()
        if "Epoch" not in df.columns:
            logger.error(f"'Epoch' column missing in training log {log_path}. Creating sample log.")
            return create_sample_log(log_path)

        return df
    except Exception as e:
        logger.error(f"Failed to load training log {log_path}: {e}")
        return create_sample_log(log_path)

def detect_metrics(df: pd.DataFrame) -> List[str]:
    metrics = []
    for col in df.columns:
        for suffix in VALID_METRIC_SUFFIXES:
            if col.startswith("Train") and col.endswith(suffix):
                metric = col.replace("Train", "")
                if f"Val{metric}" in df.columns:
                    metrics.append(metric)
    return list(set(metrics))

def plot_loss(log_path="logs/training_log.txt", output_path="logs/training_loss_plot.png"):
    try:
        df = pd.read_csv(log_path)

        if df.empty or "Epoch" not in df or "TrainLoss" not in df or "ValLoss" not in df:
            logger.warning("Training log is empty or missing required columns.")
            return

        epochs = df["Epoch"]
        train = df["TrainLoss"]
        val = df["ValLoss"]

        sm_train = moving_avg(train, 2)
        sm_val = moving_avg(val, 2)

        # Adjust epochs to match moving average length
        sm_epochs = epochs.iloc[-len(sm_train):]

        fig, ax = plt.subplots(figsize=(10, 6))  # Removed constrained_layout=True

        ax.plot(sm_epochs, sm_train, label="Train Loss", marker='o')
        ax.plot(sm_epochs, sm_val, label="Val Loss", marker='x')

        # Annotate minimum validation loss
        if len(sm_val) > 0:
            min_val_idx = np.argmin(sm_val)
            ax.annotate(
                f"Min Val: {sm_val[min_val_idx]:.4f}",
                xy=(sm_epochs.iloc[min_val_idx], sm_val[min_val_idx]),
                xytext=(sm_epochs.iloc[min_val_idx], sm_val[min_val_idx] + 0.1),
                arrowprops=dict(arrowstyle="->"),
                fontsize=9,
                color="green"
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training vs Validation Loss")
        ax.legend()
        ax.grid(True)

        # Save plot
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"✅ Loss plot saved to {output_path}")

    except Exception as e:
        logger.error(f"❌ Plotting failed: {e}")

def plot_metric(
    epochs: np.ndarray,
    train_data: np.ndarray,
    val_data: np.ndarray,
    metric_name: str,
    output_path: Path,
    window: int = DEFAULT_WINDOW,
    title: Optional[str] = None,
    annotate_extreme: bool = True,
    style: str = "seaborn-v0_8",
    colors: Dict[str, str] = None
) -> None:
    try:
        valid_style = get_valid_style(style)
        plt.style.use(valid_style)
        fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)

        sm_train = moving_avg(train_data, window)
        sm_val = moving_avg(val_data, window)
        sm_epochs = epochs[-len(sm_train):]

        colors = colors or {"train": "#1f77b4", "val": "#ff7f0e"}

        ax.plot(sm_epochs, sm_train, label=f"Train {metric_name}", marker='o', linewidth=2, markersize=6, color=colors["train"])
        ax.plot(sm_epochs, sm_val, label=f"Val {metric_name}", marker='x', linewidth=2, markersize=6, color=colors["val"])

        if annotate_extreme and len(sm_val) > 0:
            is_loss = metric_name.lower().find("loss") != -1
            extreme_idx = np.argmin(sm_val) if is_loss else np.argmax(sm_val)
            extreme_label = "Min" if is_loss else "Max"
            annotation_text = f"{extreme_label} Val: {sm_val[extreme_idx]:.4f}"
            ax.annotate(
                annotation_text,
                xy=(sm_epochs[extreme_idx], sm_val[extreme_idx]),
                xytext=(sm_epochs[extreme_idx], sm_val[extreme_idx] + (0.1 if is_loss else -0.1)),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color='black'),
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.8)
            )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_title(title or f"Training vs Validation {metric_name}", fontsize=14, pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=DEFAULT_DPI, bbox_inches='tight', format=DEFAULT_PLOT_FORMAT)
        plt.close()
        logger.info(f"{metric_name} plot saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to plot {metric_name}: {e}")
        plt.close()

def visualize_model_performance(
    log_path: Union[str, Path] = None,
    domain: Optional[str] = None,
    content: Optional[str] = None,
    window: int = DEFAULT_WINDOW,
    metrics: Optional[List[str]] = None,
    plot_format: str = DEFAULT_PLOT_FORMAT,
    style: str = "seaborn",
    colors: Dict[str, str] = None
) -> None:
    try:
        log_path = Path(log_path or BASE_LOG_DIR / "training_log.txt")

        if domain is None and DOMAIN_DETECTION_AVAILABLE and content:
            domain = detect_domain(content)
        domain = domain or "general"
        logger.info(f"Using domain: {domain}")

        df = load_training_log(log_path)
        if df.empty:
            logger.error(f"No valid data to plot in {log_path}. Using sample data.")
            return

        metrics = metrics or detect_metrics(df)
        if not metrics:
            logger.error(f"No valid metrics found in {log_path}. Using sample data.")
            return

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        output_dir = BASE_OUTPUT_DIR / domain / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)

        for metric in metrics:
            train_col = f"Train{metric}"
            val_col = f"Val{metric}"
            if train_col in df.columns and val_col in df.columns:
                output_path = output_dir / f"{metric.lower()}_plot_{timestamp}.{plot_format}"
                plot_metric(
                    epochs=df["Epoch"].values,
                    train_data=df[train_col].values,
                    val_data=df[val_col].values,
                    metric_name=metric,
                    output_path=output_path,
                    window=window,
                    title=f"{metric} Over Epochs (Domain: {domain})",
                    style=style,
                    colors=colors
                )
            else:
                logger.warning(f"Columns {train_col} and/or {val_col} not found in {log_path}.")

    except Exception as e:
        logger.error(f"Failed to visualize model performance: {e}")

if __name__ == "__main__":
    sample_content = "Explain Newton's second law of motion"
    visualize_model_performance(
        log_path=BASE_LOG_DIR / "training_log.txt",
        content=sample_content,
        window=3,
        metrics=None,
        plot_format="png",
        style="seaborn",
        colors={"train": "#1f77b4", "val": "#ff7f0e"}
    )
