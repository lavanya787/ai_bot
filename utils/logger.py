import os
import logging
import json
from datetime import datetime
from typing import Optional


class Logger:
    def __init__(self, log_dir='logs', log_level=logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        self.logs = []

        # Base logger
        self.logger = logging.getLogger("SystemLogger")
        self.logger.setLevel(self.log_level)

        # Console output
        if not self.logger.handlers:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(stream_handler)

        self._setup_logging()

    def _setup_logging(self) -> None:
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f"system_{timestamp}.log")

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            self.logger.addHandler(file_handler)

    def log_interaction(self, query: str, response: str, status: str,
                        intent: str = "", sentiment: str = "",
                        chunk_used: Optional[str] = None):
        """Logs a structured interaction to multiple formats."""
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

        # CSV Log
        csv_path = os.path.join(self.log_dir, "interactions.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", encoding="utf-8") as f:
                f.write(",".join(entry.keys()) + "\n")
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(f'"{str(value)}"' for value in entry.values()) + "\n")

        # TXT Log
        txt_path = os.path.join(self.log_dir, "interactions.txt")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write("\n=== Interaction ===\n")
            for k, v in entry.items():
                f.write(f"{k}: {v}\n")
            f.write("-" * 40 + "\n")

        # JSON Log
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

        # Intent-only log (simple text)
        if intent:
            with open(os.path.join(self.log_dir, "intent_logs.txt"), "a", encoding="utf-8") as f:
                f.write(f"{timestamp} | Intent: {intent} | Query: {query}\n")

    def log_query(self, query_type: str, query: str, dataset_info=None, response=None):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": query_type,
            "query": query,
            "dataset_info": dataset_info,
            "response": response
        }
        self.logs.append(entry)

    def get_logs(self):
        return self.logs

    def get_domain_logger(self, domain: str) -> logging.Logger:
        """Get or create a logger for a specific domain."""
        domain_log_file = os.path.join(self.log_dir, f"{domain}.log")
        domain_logger = logging.getLogger(f"Domain_{domain}")

        if not domain_logger.handlers:
            domain_logger.setLevel(self.log_level)
            file_handler = logging.FileHandler(domain_log_file)
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            domain_logger.addHandler(file_handler)

        return domain_logger

    def log_info(self, message: str, exc_info: bool = False):
        self.logger.info(message, exc_info=exc_info)

    def log_warning(self, message: str, exc_info: bool = False):
        self.logger.warning(message, exc_info=exc_info)

    def log_error(self, message: str, exc_info: bool = False):
        self.logger.error(message, exc_info=exc_info)

    def log_debug(self, message: str, domain: Optional[str] = None):
        logger = self.get_domain_logger(domain) if domain else self.logger
        logger.debug(message)
