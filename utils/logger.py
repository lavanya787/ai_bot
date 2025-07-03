import os
import logging
import json
from datetime import datetime
from typing import Optional
import psutil
import GPUtil

class Logger:
    _shared_logger = None  # class-level shared logger

    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        self.logs = []

        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(self.log_level)

        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(stream_handler)

        self._setup_logging()

        # Set shared logger if not already set
        if Logger._shared_logger is None:
            Logger._shared_logger = self.logger

    def _setup_logging(self) -> None:
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"system_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)

    @classmethod
    def get_logger(cls):
        if cls._shared_logger is None:
            cls()
        return cls._shared_logger

    def log_interaction(self, query: str, response: str, status: str,
                        intent: str = "", sentiment: str = "",
                        chunk_used: Optional[str] = None):
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "Time": timestamp,
            "Query": query,
            "Intent": intent or "N/A",
            "Sentiment": sentiment or "N/A",
            "Status": status,
            "ChunkUsed": chunk_used or "N/A",
            "Response": response
        }

        # CSV log
        csv_path = os.path.join(self.log_dir, "interactions.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(entry.keys()) + "\n")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(f'"{str(value)}"' for value in entry.values()) + "\n")

        # TXT log
        txt_path = os.path.join(self.log_dir, "interactions.txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n=== Interaction ===\n")
            for k, v in entry.items():
                f.write(f"{k}: {v}\n")
            f.write("-" * 40 + "\n")

        # JSON log
        json_path = os.path.join(self.log_dir, "interactions.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as jf:
                try:
                    logs = json.load(jf)
                except json.JSONDecodeError:
                    logs = []
        else:
            logs = []
        logs.append(entry)
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(logs, jf, indent=2)

        # Intent log
        if intent:
            with open(os.path.join(self.log_dir, "intent_logs.txt"), "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | Intent: {intent} | Query: {query}\n")

    def log_info(self, message: str, exc_info: bool = False):
        self.logger.info(message, exc_info=exc_info)

    def log_warning(self, message: str, exc_info: bool = False):
        self.logger.warning(message, exc_info=exc_info)

    def log_error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str, domain: Optional[str] = None):
        logger = self.logger if domain is None else self.get_domain_logger(domain)
        logger.debug(message)

    def get_domain_logger(self, domain: str) -> logging.Logger:
        domain_log_file = os.path.join(self.log_dir, f"{domain}.log")
        domain_logger = logging.getLogger(f"Domain_{domain}")

        if not domain_logger.handlers:
            domain_logger.setLevel(self.log_level)
            file_handler = logging.FileHandler(domain_log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            domain_logger.addHandler(file_handler)

        return domain_logger


# System stats utilities
def format_memory_stats():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    stats = f"🖥️ | CPU: {cpu:.1f}% | RAM: {mem.percent:.2f}% used ({mem.used / 1024**2:.2f} MB)"
    return stats


def log_gpu_stats(logger=None):
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return

        for gpu in gpus:
            msg = (
                f"🧠 GPU: {gpu.name} | "
                f"Load: {gpu.load * 100:.1f}% | "
                f"Free Mem: {gpu.memoryFree:.1f}MB | "
                f"Used Mem: {gpu.memoryUsed:.1f}MB | "
                f"Total Mem: {gpu.memoryTotal:.1f}MB"
            )
            if logger:
                logger.info(msg)
            else:
                print(msg)
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Failed to log GPU stats: {e}")
        else:
            print(f"⚠️ Failed to log GPU stats: {e}")


def log_disk_io():
    logger = Logger.get_logger()
    io_counters = psutil.disk_io_counters()
    read_mb = io_counters.read_bytes / (1024 * 1024)
    write_mb = io_counters.write_bytes / (1024 * 1024)
    logger.info(f"💾 Disk I/O | Read: {read_mb:.2f} MB | Write: {write_mb:.2f} MB")


def get_file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0
# utils/logger.py

logger = Logger.get_logger()
