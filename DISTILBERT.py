# ============================================================
# MÔ HÌNH 1: DISTILBERT 
# Dataset: Tweets.csv (airline sentiment)
# ============================================================

import pandas as pd
import numpy as np
import re
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
# from google.colab import files # Uncomment if running standalone in Colab
import os

import seaborn as sns
import matplotlib.pyplot as plt

# Check if 'Tweets.csv' already exists to avoid re-uploading
# if not os.path.exists('Tweets.csv'): # Uncomment if running standalone in Colab
#     uploaded = files.upload() # Uncomment if running standalone in Colab
#     for fn in uploaded.keys(): # Uncomment if running standalone in Colab
#         print(f'User uploaded file "{fn}" with length {len(uploaded[fn])} bytes') # Uncomment if running standalone in Colab
# else: # Uncomment if running standalone in Colab
#     print("'Tweets.csv' already exists. Skipping upload.") # Uncomment if running standalone in Colab

# Kiểm tra GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang dùng: {device.upper()}")

df = pd.read_csv("Tweets.csv")
df = df[["text", "airline_sentiment"]].copy()
print("Phân bố nhãn:")
print(df["airline_sentiment"].value_counts())

label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(label_map)
df = df[["text", "label"]].dropna()  # Bỏ dòng NaN nếu có
# df.head() # Uncomment to display head

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)
print("Dữ liệu sau khi làm sạch:")
# display(df.head()) # Uncomment to display head

# === CODE ADDED FOR BALANCING ===
def balance_dataframe(df_input):
    # Determine the count of the smallest class
    min_class_count = df_input['label'].value_counts().min()

    # Create balanced dataframes for each class
    df_negative = df_input[df_input['label'] == 0].sample(min_class_count, random_state=42)
    df_neutral = df_input[df_input['label'] == 1].sample(min_class_count, random_state=42)
    df_positive = df_input[df_input['label'] == 2].sample(min_class_count, random_state=42)

    # Concatenate them into a new balanced dataframe
    df_balanced = pd.concat([df_negative, df_neutral, df_positive]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Dữ liệu sau khi cân bằng: {len(df_balanced)} mẫu")
    print("Phân bố nhãn sau cân bằng:")
    print(df_balanced['label'].value_counts())
    return df_balanced

df = balance_dataframe(df)
# === END ADDED CODE ===

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),  # Giữ tỉ lệ nhãn đồng đều
)
print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

MODEL_NAME = "distilbert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding=True,
    max_length=128,
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=128,
)
print("Tokenize xong!")

class TwitterDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# FIX: Dòng này trước bị thụt vào trong class → IndentationError
train_dataset = TwitterDataset(train_encodings, train_labels)
test_dataset  = TwitterDataset(test_encodings,  test_labels)
print(f"Train dataset: {len(train_dataset)} mẫu | Test dataset: {len(test_dataset)} mẫu")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)
model = model.to(device)  # Đẩy lên GPU nếu có
print("Tải mô hình xong!")

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,                  # Tăng lên 3 cho độ chính xác tốt hơn
    per_device_train_batch_size=32,      # Tăng batch (GPU T4 chịu được)
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,         # Tự lấy model tốt nhất
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),      # Bật FP16 trên GPU → nhanh ~2x
    dataloader_num_workers=2,            # Parallel data loading
    report_to="none",                    # Tắt WandB để không bị hỏi login
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    return {"accuracy": acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer # Added tokenizer for proper saving/loading
)

trainer.train()

distilbert_eval_results = trainer.evaluate()
print("\nKết quả đánh giá DistilBERT:")
print(distilbert_eval_results)

# Store DistilBERT accuracy and predictions in distinct global variables
distilbert_acc = distilbert_eval_results['eval_accuracy']
preds_output_distilbert = trainer.predict(test_dataset)
distilbert_y_pred = preds_output_distilbert.predictions.argmax(-1)

print("\nClassification Report DistilBERT:")
print(classification_report(test_labels, distilbert_y_pred, target_names=["Negative","Neutral","Positive"])) # label_names_list should be defined

# Plotting the confusion matrix
cm = confusion_matrix(test_labels, distilbert_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=["Negative","Neutral","Positive"],
    yticklabels=["Negative","Neutral","Positive"],
)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix DistilBERT')
plt.show()

label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text: str) -> str:
    """Dự đoán cảm xúc cho một đoạn text."""
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # FIX: đưa input lên cùng device với model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_names[prediction]

# Test thử
samples = [
    "this airline is amazing",
    "terrible service, lost my luggage",
    "flight was ok, nothing special",
]
for s in samples:
    print(f"'{s}' → {predict_sentiment(s)}")

model.save_pretrained("twitter_sentiment_model_distilbert")
tokenizer.save_pretrained("twitter_sentiment_model_distilbert")

# !zip -r twitter_sentiment_model_distilbert.zip twitter_sentiment_model_distilbert # Uncomment if running standalone in Colab

# from google.colab import files # Uncomment if running standalone in Colab
# files.download("twitter_sentiment_model_distilbert.zip") # Uncomment if running standalone in Colab
print("Tải về xong!")