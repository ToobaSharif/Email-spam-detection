# 📧 Email Spam Detection — BERT Model Trainer

This repository contains a Python script (`train.py`) that fine-tunes the [BERT (base-uncased)](https://huggingface.co/bert-base-uncased) model to classify email subjects as **spam** or **ham** using a labeled dataset in Excel format.

---

## 📁 Project Structure

├── data/
│ └── Email-DataSet.xlsx # Input Excel dataset with 'Subject' and 'Spam/Ham' columns
├── model/ # Output directory for the saved model and tokenizer
├── logs/ # Logs from the training process
├── results/ # Evaluation and checkpoint results
└── train.py # Main script to train the model

---

 What It Does

- Loads and cleans a labeled email dataset (`Spam` / `Ham`)
- Tokenizes subjects using BERT tokenizer
- Trains a `BertForSequenceClassification` model using Hugging Face's `Trainer`
- Evaluates and saves the best-performing model



Dependencies

Make sure you have the following Python libraries installed:


pip install pandas scikit-learn torch transformers openpyxl

How to Run
Place your Excel dataset in the ./data folder

Required columns: "Subject" and "Spam/Ham" (labels must be "spam" or "ham")

Run the script:

bash
Copy
Edit
python train.py
Model and tokenizer will be saved in the ./model directory after training.

 Training Details
Model Used: BERT-base-uncased (from Hugging Face)

Epochs: 2 (modifiable)

Batch Size: 16

Evaluation Strategy: Per epoch

Token Limit: 64 tokens max per subject

GPU Support: Yes (automatically used if available)

 Output
After successful training:

Trained BERT model files (pytorch_model.bin, config.json, etc.)

How to Run
Place your Excel dataset in the ./data folder

Required columns: "Subject" and "Spam/Ham" (labels must be "spam" or "ham")

Run the script:

bash
Copy
Edit
python train.py
Model and tokenizer will be saved in the ./model directory after training.

📊 Training Details
Model Used: BERT-base-uncased (from Hugging Face)

Epochs: 2 (modifiable)

Batch Size: 16

Evaluation Strategy: Per epoch

Token Limit: 64 tokens max per subject

GPU Support: Yes (automatically used if available)

📁 Output
After successful training:

Trained BERT model files (pytorch_model.bin, config.json, etc.)

Tokenizer files (vocab.txt, tokenizer_config.json, etc.)

These are saved in the /model folder and can be loaded later for inference or API integration.

 Example Dataset Format
Subject	Spam/Ham
"You’ve won a free iPhone!"	spam
"Project update meeting at 3pm"	ham



Tokenizer files (vocab.txt, tokenizer_config.json, etc.)

These are saved in the /model folder and can be loaded later for inference or API integration.

 Example Dataset Format
Subject	Spam/Ham
"You’ve won a free iPhone!"	spam
"Project update meeting at 3pm"	ham


