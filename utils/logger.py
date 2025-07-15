import os
import logging
import json
from datetime import datetime
from typing import Optional
import psutil
import GPUtil

class Logger:
    _instance = None  # Singleton instance

    def __new__(cls, log_dir='logs', log_level=logging.INFO):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._setup(log_dir, log_level)
        return cls._instance

    def _setup(self, log_dir, log_level):
        self.log_dir = log_dir
        self.log_level = log_level
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("AIAppLogger")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # Avoid duplicate logs

        log_file = os.path.join(log_dir, "main.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        # File Handler (write mode so it doesn’t grow endlessly unless rotated)
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # Stream (console) Handler
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

    def get_logger(self):
        return self.logger

    def log_interaction(self, query: str, response: str, status: str,
                        intent: str = "", sentiment: str = "", chunk_used: Optional[str] = None):
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
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(entry.keys()) + "\n")
            f.write(",".join(f'"{str(v)}"' for v in entry.values()) + "\n")

        # JSON log
        json_path = os.path.join(self.log_dir, "interactions.json")
        logs = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    logs = json.load(f)
            except json.JSONDecodeError:
                logs = []
        logs.append(entry)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2)

        # TXT log (human-readable)
        with open(os.path.join(self.log_dir, "interactions.txt"), "a", encoding="utf-8") as f:
            f.write("\n=== Interaction ===\n")
            for k, v in entry.items():
                f.write(f"{k}: {v}\n")
            f.write("-" * 40 + "\n")

        # Intent-only log
        if intent:
            with open(os.path.join(self.log_dir, "intent_logs.txt"), "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | Intent: {intent} | Query: {query}\n")

    def log_info(self, message: str, exc_info=False):
        self.logger.info(message, exc_info=exc_info)

    def log_warning(self, message: str, exc_info=False):
        self.logger.warning(message, exc_info=exc_info)

    def log_error(self, message: str, exc_info=False):
        self.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str):
        self.logger.debug(message)

    def get_domain_logger(self, domain: str) -> logging.Logger:
        domain_logger = logging.getLogger(f"DomainLogger_{domain}")
        if not domain_logger.handlers:
            domain_logger.setLevel(self.log_level)
            file_path = os.path.join(self.log_dir, f"{domain}.log")
            handler = logging.FileHandler(file_path, encoding="utf-8")
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            domain_logger.addHandler(handler)
        return domain_logger


# System Monitoring Utilities
def format_memory_stats():
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory()
    return f"🖥️ CPU: {cpu:.1f}% | RAM: {mem.percent:.1f}% ({mem.used / 1024**3:.2f} GB used)"

def log_gpu_stats(logger=None):
    try:
        gpus = GPUtil.getGPUs()
        if not gpus:
            return
        for gpu in gpus:
            msg = (f"🧠 GPU {gpu.name} | Load: {gpu.load * 100:.1f}% | "
                   f"Used: {gpu.memoryUsed:.1f}MB / {gpu.memoryTotal:.1f}MB")
            logger.info(msg) if logger else print(msg)
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Failed to log GPU stats: {e}")
        else:
            print(f"⚠️ GPU log failed: {e}")

def log_disk_io():
    logger = Logger().get_logger()
    io = psutil.disk_io_counters()
    logger.info(f"💾 Disk I/O | Read: {io.read_bytes / (1024**2):.2f} MB | Write: {io.write_bytes / (1024**2):.2f} MB")

def get_file_size_mb(path):
    return os.path.getsize(path) / (1024**2) if os.path.exists(path) else 0.0


# Global access
logger = Logger().get_logger()
