import json
import os
from utils.logger import Logger, format_memory_stats, log_gpu_stats, log_disk_io, get_file_size_mb

class Memory:
    def __init__(self, storage_file="conversation_history.json"):
        self.logger = Logger().get_logger()
        self.history = []
        self.storage_file = storage_file
        # Use direct logging.Logger methods as fallback
        self.logger.info(f"Initializing Memory with storage file: {storage_file}")  # Fallback to info
        log_gpu_stats(self.logger)
        self.logger.info(format_memory_stats())  # Fallback to info
        log_disk_io()
        self._load_history()
        self.chat_logs = {}


    def add_message(self, session_id, role, message):
        if session_id not in self.chat_logs:
            self.chat_logs[session_id] = []
        self.chat_logs[session_id].append({"role": role, "message": message})

    def get_context(self, session_id, limit=None):
        """
        Retrieve recent chat messages for the given session.
        If limit is None, return all messages. Otherwise return the last `limit` messages.
        """
        messages = self.chat_logs.get(session_id, [])
        return messages if limit is None else messages[-limit:]
    
    def _load_history(self):
        self.logger.info(f"Loading history from {self.storage_file}")  # Fallback to info
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                self.logger.info(f"Loaded {len(self.history)} entries from {self.storage_file}")  # Fallback to info
                file_size = get_file_size_mb(self.storage_file)
                self.logger.info(f"File size: {file_size:.2f} MB")  # Fallback to info
            else:
                self.logger.info(f"No existing history file found at {self.storage_file}")  # Fallback to info
        except json.JSONDecodeError:
            self.logger.warning("Corrupted history file, starting fresh")  # Fallback to warning
            self.history = []
        except Exception as e:
            self.logger.error(f"Error loading history: {e}")  # Fallback to error
            self.history = []

    def _save_history(self):
        self.logger.info(f"Saving history to {self.storage_file}")  # Fallback to info
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
            file_size = get_file_size_mb(self.storage_file)
            self.logger.info(f"History saved, file size: {file_size:.2f} MB")  # Fallback to info
        except Exception as e:
            self.logger.error(f"Error saving history: {e}")  # Fallback to error

    def append(self, role, message):
        self.logger.info(f"Appending message - Role: {role}, Message: {message[:50]}{'...' if len(message) > 50 else ''}")  # Fallback to info
        self.history.append({"role": role, "message": message})
        self._save_history()
        log_gpu_stats(self.logger)
        self.logger.info(format_memory_stats())  # Fallback to info

    def get_formatted(self):
        self.logger.info(f"Retrieving formatted history with {len(self.history)} entries")  # Fallback to info
        return "\n".join(f"{entry['role']}: {entry['message']}" for entry in self.history)

    def clear(self):
        self.logger.info("Clearing history")  # Fallback to info
        self.history = []
        self._save_history()
        log_gpu_stats(self.logger)
        self.logger.info(format_memory_stats())  # Fallback to info