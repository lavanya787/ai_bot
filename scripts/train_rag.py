import os
import argparse
import logging
from llm_handler import LLMHandler
from utils.visualizer import plot_loss
from utils.logger import Logger
from utils.text_utils import load_documents_from_folder  # (Optional: if extracted to utils)
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,  # Ensure it's not default stderr
    encoding="utf-8",   # ✅ Add this to allow emojis and Unicode
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(data_dir, epochs, batch_size, save_path, log_path, plot_path):
    logger.info(f"📂 Loading documents from: {data_dir}")
    documents = load_documents_from_folder(data_dir)

    if not documents:
        logger.error("❌ No valid documents found.")
        return

    handler = LLMHandler()

    logger.info("📥 Indexing documents...")
    for fname, content in documents.items():
        handler.index_document(fname, content)

    logger.info("🚀 Training RAG model ...")
    handler.train_on_documents(
        epochs=epochs,
        batch_size=batch_size,
        save_path=save_path,
        log_path=log_path
    )

    logger.info("📊 Generating loss plot...")
    plot_loss(log_path=log_path, output_path=plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline trainer for custom RAG model")
    parser.add_argument("--data_dir", type=str, default="rag_data", help="Directory with .txt/.csv documents")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="checkpoint.pt", help="Path to save training checkpoint")
    parser.add_argument("--log_path", type=str, default="logs/training_log.txt", help="Path to write training log")
    parser.add_argument("--plot_path", type=str, default="logs/training_loss_plot.png", help="Path to save loss plot")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    os.makedirs(os.path.dirname(args.plot_path), exist_ok=True)

    main(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_path=args.save_path,
        log_path=args.log_path,
        plot_path=args.plot_path
    )
