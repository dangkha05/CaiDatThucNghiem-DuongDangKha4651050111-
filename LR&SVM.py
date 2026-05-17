
# ============================================================
# MÔ HÌNH 2: TF-IDF + Logistic Regression + SVM
# So sánh với Mô hình 1: DistilBERT (Deep Learning)
# Dataset: Tweets.csv (airline sentiment)
# ============================================================

import pandas as pd
import numpy as np
import re
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# Load dữ liệu
# This part assumes 'Tweets.csv' is already available or will be uploaded.
# from google.colab import files
# uploaded = files.upload() # Uncomment if running standalone in Colab and file needs upload

df = pd.read_csv("Tweets.csv")
df = df[["text", "airline_sentiment"]].copy()

label_map = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["airline_sentiment"].map(label_map)
df = df[["text", "label"]].dropna()

# ============================================================
# Clean text
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text"] = df["text"].apply(clean_text)

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
# Train/Test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["text"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist(),
)

# ============================================================
# Logistic Regression
lr_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        random_state=42,
    )),
])

lr_pipeline.fit(train_texts, train_labels)
lr_preds = lr_pipeline.predict(test_texts)
lr_acc = accuracy_score(test_labels, lr_preds)

print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

# ============================================================
# SVM
svm_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )),
    ("clf", LinearSVC(
        C=0.5,
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )),
])

svm_pipeline.fit(train_texts, train_labels)
svm_preds = svm_pipeline.predict(test_texts)
svm_acc = accuracy_score(test_labels, svm_preds)

print(f"SVM Accuracy: {svm_acc:.4f}")

# ============================================================
# Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
label_names = ["Negative", "Neutral", "Positive"]

for ax, preds, title in zip(
    axes,
    [lr_preds, svm_preds],
    ["Logistic Regression", "SVM (LinearSVC)"],
):
    cm = confusion_matrix(test_labels, preds)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix — {title}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
plt.savefig("confusion_matrix_ml.png", dpi=150)
plt.show()

# ============================================================
# Lưu model
best_model = svm_pipeline if svm_acc >= lr_acc else lr_pipeline

joblib.dump(best_model, "traditional_ml_model.pkl")

print("Hoàn tất huấn luyện và lưu model.")