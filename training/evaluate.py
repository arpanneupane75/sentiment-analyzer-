from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import numpy as np
import matplotlib.pyplot as plt


MODEL_PATH = "./model/sentiment_model"

MAX_LENGTH = 256
TEST_SIZE = 1000
SEED = 42


print("Loading IMDb test dataset...")

dataset = load_dataset("imdb")
test_dataset = dataset["test"]

print(f"Original test samples: {len(test_dataset)}")


# Shuffle first so both classes are represented
test_dataset = (
    test_dataset
    .shuffle(seed=SEED)
    .select(range(TEST_SIZE))
)

print(f"Evaluation samples: {len(test_dataset)}")


# Check class distribution
labels = test_dataset["label"]

negative_count = labels.count(0)
positive_count = labels.count(1)

print("\nTest label distribution:")
print(f"Negative: {negative_count}")
print(f"Positive: {positive_count}")


# Tokenizer
print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)


def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


print("Tokenizing test dataset...")

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True,
)


# Model
print("\nLoading trained model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)


trainer = Trainer(
    model=model,
    processing_class=tokenizer,
)


# Predictions
print("\nRunning predictions...")

predictions = trainer.predict(test_dataset)

predicted_labels = np.argmax(
    predictions.predictions,
    axis=-1,
)

true_labels = predictions.label_ids


# Metrics
accuracy = accuracy_score(
    true_labels,
    predicted_labels,
)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="binary",
        zero_division=0,
    )
)


print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# Classification report
print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        true_labels,
        predicted_labels,
        target_names=[
            "Negative",
            "Positive",
        ],
        zero_division=0,
    )
)


# Confusion matrix
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
)

print("========================================")
print("CONFUSION MATRIX")
print("========================================")

print(matrix)


# ============================================
# Create confusion matrix visualization
# ============================================

fig, ax = plt.subplots(figsize=(7, 6))

image = ax.imshow(matrix)

ax.set_title("IMDb Sentiment Classification - Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["Negative", "Positive"])
ax.set_yticklabels(["Negative", "Positive"])


for i in range(2):
    for j in range(2):
        ax.text(
            j,
            i,
            matrix[i, j],
            ha="center",
            va="center",
            fontsize=14,
        )


plt.colorbar(image)

plt.tight_layout()

plt.savefig(
    "evaluation_results.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print("\nSaved confusion matrix to:")
print("evaluation_results.png")