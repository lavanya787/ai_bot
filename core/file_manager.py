import os
import json
import torch
from file_processing.processor import load_file_text
from file_processing.chunker import chunk_text
from retrieval.meta_retriever import MetaRetriever

class FileManager:
    BASE_LOG_DIR = "logs"

    def get_log_dir(self, filename):
        """
        Create and return a log directory based on the filename.
        Args:
            filename (str): The name of the file.
        Returns:
            str: The path to the log directory.
        """
        tag = os.path.splitext(os.path.basename(filename))[0]
        path = os.path.join(self.BASE_LOG_DIR, tag)
        os.makedirs(path, exist_ok=True)
        return path

    def save_file(self, uploaded_file, destination_path):
        """
        Save an uploaded file to the specified destination path.
        Args:
            uploaded_file: The uploaded file object (e.g., from Streamlit file_uploader).
            destination_path (str): The path where the file should be saved.
        Returns:
            tuple: (file_path, log_dir) - The saved file path and the log directory.
        """
        log_dir = self.get_log_dir(uploaded_file.name)
        file_path = destination_path  # Use the provided destination path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        return file_path, log_dir

    def process_file_chunks(self, filepath, log_dir):
        """
        Process a file into chunks and save the chunks to a JSON file.
        Args:
            filepath (str): The path to the file to process.
            log_dir (str): The log directory to save the chunks JSON.
        Returns:
            list: The list of text chunks.
        """
        text = load_file_text(filepath)
        chunks = chunk_text(text)
        chunk_path = os.path.join(log_dir, f"{os.path.basename(filepath)}_chunks.json")
        with open(chunk_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        return chunks

    def get_retriever(self, chunks):
        """
        Create a MetaRetriever instance for the given chunks.
        Args:
            chunks (list): The list of text chunks.
        Returns:
            MetaRetriever: The retriever instance.
        """
        return MetaRetriever(chunks)

    def save_model_with_tag(self, model, stoi, itos, log_dir, tag):
        """
        Save a model and its vocabulary with a specific tag.
        Args:
            model: The model to save.
            stoi (dict): The string-to-index vocabulary mapping.
            itos (dict): The index-to-string vocabulary mapping.
            log_dir (str): The log directory to save the model and vocab.
            tag (str): The tag to use for the saved files.
        """
        torch.save(model.state_dict(), os.path.join(log_dir, f"{tag}_model.pth"))
        with open(os.path.join(log_dir, f"{tag}_vocab.json"), "w") as f:
            json.dump({"stoi": stoi, "itos": itos}, f, indent=2)

# Create an instance of FileManager for use in imports
file_manager = FileManager()