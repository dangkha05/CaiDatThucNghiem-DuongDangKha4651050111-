
# ============================================================
# MÔ HÌNH 3: BERT (bert-base-uncased)
# Dataset: Tweets.csv (airline sentiment)
# ============================================================

# Bước 1 — Cài thư viện
# !pip install -q transformers accelerate scikit-learn pandas # Uncomment if running standalone in Colab

# ============================================================
# Bước 2 — Upload dữ liệu
# from google.colab import files
# uploaded = files.upload()  # Chọn Tweets.csv # Uncomment if running standalone in Colab

# ============================================================
# Bước 3 — Import
import pandas as pd
import numpy as np
import re
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Đang dùng: {device.upper()}")

# ============================================================
# Bước 4 — Load & tiền xử lý (giống hệt mô hình 1 & 2)
df = pd.read_csv("Tweets.csv")
df = df[["text", "airline_sentiment"]].copy()
print("Phân bố nhãn:")
print(df["airline_sentiment"].value_counts())

label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(label_map)
df = df[["text", "label"]].dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
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

# ============================================================
# Bước 5 — Train/Test split (cùng random_state=42)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),
)
print(f"Train: {len(train_texts)} | Test: {len(test_texts)}")

# ============================================================
# Bước 6 — Tokenize
# BERT dùng bert-base-uncased thay vì distilbert
MODEL_NAME = "bert-base-uncased"

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

# ============================================================
# Bước 7 — Dataset class
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

train_dataset = TwitterDataset(train_encodings, train_labels)
test_dataset  = TwitterDataset(test_encodings,  test_labels)
print(f"Train dataset: {len(train_dataset)} mẫu | Test dataset: {len(test_dataset)} mẫu")

# ============================================================
# Bước 8 — Load BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)
model = model.to(device)
print("Tải BERT xong!")

# ============================================================
# Bước 9 — TrainingArguments
# BERT nặng hơn DistilBERT → batch nhỏ hơn một chút để tránh OOM
training_args = TrainingArguments(
    output_dir="./results_bert",
    num_train_epochs=3,
    per_device_train_batch_size=16,      # Nhỏ hơn DistilBERT vì BERT nặng hơn
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    logging_dir="./logs_bert",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),      # FP16 → nhanh ~2x trên GPU
    dataloader_num_workers=2,
    report_to="none",
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    acc    = accuracy_score(labels, preds)
    return {"accuracy": acc}

# ============================================================
# Bước 10 — Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer, # Added tokenizer for proper saving/loading
)

trainer.train()

# Store BERT accuracy and predictions in distinct global variables
bert_eval_results = trainer.evaluate()
bert_acc = bert_eval_results["eval_accuracy"]
preds_output_bert = trainer.predict(test_dataset)
bert_y_pred = preds_output_bert.predictions.argmax(-1)

print(f"\nBERT Accuracy: {bert_acc:.4f}")

print("\nClassification Report (BERT):")
print(classification_report(test_labels, bert_y_pred, target_names=["Negative", "Neutral", "Positive"])) # label_names_list should be defined

# ============================================================
# Bước 12 — Confusion Matrix
label_names_list = ["Negative", "Neutral", "Positive"]
cm = confusion_matrix(test_labels, bert_y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
            xticklabels=label_names_list,
            yticklabels=label_names_list)
plt.title("Confusion Matrix — BERT")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_bert.png", dpi=150)
plt.show()
print("Đã lưu confusion_matrix_bert.png")

# Removed the comparison table from here as it's handled in a separate comparison cell

# ============================================================
# Bước 14 — Inference
label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}

def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return label_names[prediction]

samples = [
    "this airline is amazing",
    "terrible service, lost my luggage",
    "flight was ok, nothing special",
]
print("\nTest inference:")
for s in samples:
    print(f"  '{s}' → {predict_sentiment(s)}")

# ============================================================
# Bước 15 — Lưu model
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_model")

# Lưu file .pth để dùng với Streamlit
torch.save({
    "model_state_dict": model.state_dict(),
    "label_names": label_names,
    "model_name": MODEL_NAME,
    "num_labels": 3,
}, "bert_sentiment_model.pth")

# !zip -r bert_sentiment_model.zip bert_sentiment_model # Uncomment if running standalone in Colab

# from google.colab import files # Uncomment if running standalone in Colab
# files.download("bert_sentiment_model.pth") # Uncomment if running standalone in Colab
# files.download("bert_sentiment_model.zip") # Uncomment if running standalone in Colab
print("Tải về xong!")