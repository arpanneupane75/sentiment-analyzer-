from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ============================================
# Configuration
# ============================================

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./model/sentiment_model"

MAX_LENGTH = 256

# CPU-friendly experiment
TRAIN_SIZE = 2000
VALIDATION_SIZE = 500

SEED = 42


# ============================================
# 1. Load IMDb dataset
# ============================================

print("Loading IMDb dataset...")

dataset = load_dataset("imdb")

train_dataset = dataset["train"]
test_dataset = dataset["test"]


print(f"Original training samples: {len(train_dataset)}")
print(f"Original test samples: {len(test_dataset)}")


# ============================================
# 2. Create validation split
# ============================================

split_dataset = train_dataset.train_test_split(
    test_size=0.1,
    seed=SEED,
)

train_dataset = split_dataset["train"]
validation_dataset = split_dataset["test"]


# ============================================
# 3. Shuffle before selecting subset
# ============================================

train_dataset = (
    train_dataset
    .shuffle(seed=SEED)
    .select(range(TRAIN_SIZE))
)

validation_dataset = (
    validation_dataset
    .shuffle(seed=SEED)
    .select(range(VALIDATION_SIZE))
)


print(f"Training samples used: {len(train_dataset)}")
print(f"Validation samples used: {len(validation_dataset)}")


# ============================================
# 4. Load tokenizer
# ============================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================
# 5. Tokenization
# ============================================

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


print("Tokenizing training data...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
)


print("Tokenizing validation data...")

validation_dataset = validation_dataset.map(
    tokenize_function,
    batched=True,
)


# ============================================
# 6. Load model
# ============================================

print("\nLoading DistilBERT model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)


# ============================================
# 7. Metrics
# ============================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="binary",
            zero_division=0,
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================
# 8. Training configuration
# ============================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,

    report_to="none",

    seed=SEED,
)


# ============================================
# 9. Trainer
# ============================================

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=validation_dataset,

    processing_class=tokenizer,

    compute_metrics=compute_metrics,
)


# ============================================
# 10. Train
# ============================================

print("\n========================================")
print("Starting model training...")
print("========================================\n")

trainer.train()


# ============================================
# 11. Validation evaluation
# ============================================

print("\n========================================")
print("Validation Results")
print("========================================")

validation_results = trainer.evaluate(
    validation_dataset
)

for key, value in validation_results.items():

    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")


# ============================================
# 12. Save model
# ============================================

print("\nSaving model...")

trainer.save_model(
    OUTPUT_DIR
)

tokenizer.save_pretrained(
    OUTPUT_DIR
)


print(f"\nModel saved to: {OUTPUT_DIR}")

print("\nTraining completed successfully.")