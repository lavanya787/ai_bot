import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset
from utils.logger import Logger

class QueryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def fine_tune_distilbert(data_path: str, output_dir: str = "models/distilbert"):
    logger = Logger()
    logger.log_info("Starting DistilBERT fine-tuning")
    
    try:
        # Load dataset (assumed CSV with 'query' and 'intent' columns)
        df = pd.read_csv(data_path)
        texts = df['query'].values
        labels = df['intent'].values
        
        # Encode labels
        le = LabelEncoder()
        encoded_labels = le.fit_transform(labels)
        
        # Initialize tokenizer and model
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=len(le.classes_)
        )
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = QueryDataset(train_texts, train_labels, tokenizer)
        val_dataset = QueryDataset(val_texts, val_labels, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='logs/distilbert',
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.log_info(f"DistilBERT fine-tuning completed. Model saved to {output_dir}")
        
    except Exception as e:
        logger.log_error(f"DistilBERT fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    fine_tune_distilbert("data/intent_dataset.csv")