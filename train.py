# train.py

import pandas as pd
import torch
import os
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# --- 1. Configuration ---
DATA_FILE_PATH = './data/Email-DataSet.xlsx'
MODEL_SAVE_PATH = './model'

# --- 2. Custom Dataset Class ---
class EmailDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# --- 3. Main Training Function ---
def train_model():
    print("--- Starting Model Training ---")

    # --- Load and Preprocess Data ---
    print(f"Loading dataset from {DATA_FILE_PATH}...")
    df = pd.read_excel(DATA_FILE_PATH, sheet_name="Sheet1")
    print(f"Initial Dataset Size: {df.shape}")

    df = df.dropna(subset=["Subject", "Spam/Ham"]).copy()
    df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['label']).copy()
    df['label'] = df['label'].astype(int)
    df['Subject'] = df['Subject'].astype(str)
    
    if df.empty:
        print("Error: Dataset is empty after preprocessing. Halting training.")
        return

    print(f"Dataset Size after cleaning: {df.shape}")

    # --- Split Data ---
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Subject'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42, stratify=df['label'].tolist()
    )

    # --- Tokenization ---
    print("Loading BERT tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    print("Tokenizing training and validation data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

    train_dataset = EmailDataset(train_encodings, train_labels)
    val_dataset = EmailDataset(val_encodings, val_labels)

    # --- Model Loading ---
    print("Loading pre-trained BERT model for sequence classification...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # Check if a GPU is available and move the model to the GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model will be trained on: {device}")

    # --- Training ---
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=2,  # Increased epochs for better performance
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # Important for saving the best version
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Starting trainer...")
    trainer.train()

    # --- Save Model and Tokenizer ---
    print(f"Training complete. Saving model and tokenizer to {MODEL_SAVE_PATH}...")
    
    # Ensure the save directory exists
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)
    
    print("--- Model Training Finished Successfully ---")


if __name__ == "__main__":
    train_model()