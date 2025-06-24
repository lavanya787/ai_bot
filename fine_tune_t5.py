import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from utils.logger import Logger

class ResponseDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_input_length=512, max_target_length=150):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_text = str(self.inputs[idx])
        target_text = str(self.targets[idx])
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

def fine_tune_t5(data_path: str, output_dir: str = "models/t5"):
    logger = Logger()
    logger.log_info("Starting T5 fine-tuning")
    
    try:
        # Load dataset (assumed CSV with 'context' and 'response' columns)
        df = pd.read_csv(data_path)
        inputs = df['context'].values
        targets = df['response'].values
        
        # Initialize tokenizer and model
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        
        # Split data
        train_inputs, val_inputs, train_targets, val_targets = train_test_split(
            inputs, targets, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = ResponseDataset(train_inputs, train_targets, tokenizer)
        val_dataset = ResponseDataset(val_inputs, val_targets, tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='logs/t5',
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
        logger.log_info(f"T5 fine-tuning completed. Model saved to {output_dir}")
        
    except Exception as e:
        logger.log_error(f"T5 fine-tuning failed: {str(e)}")
        raise

if __name__ == "__main__":
    fine_tune_t5("data/response_dataset.csv")