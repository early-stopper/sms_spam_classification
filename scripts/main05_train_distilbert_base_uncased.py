
import numpy as np
import pandas as pd
import sklearn.metrics
import transformers
import datasets

import source.io

def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    labels = pred.label_ids
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(labels, preds, average='binary')
    accuracy = sklearn.metrics.accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

def main():
    path_data = source.io.path_root/"data/splits/2025-06-19"
    train, test = [pd.read_csv(path_data / f"{split}.csv", index_col=False) for split in ['train', 'test']]

    # Hugging Face expects a column named 'text'
    train = train.rename(columns={"message": "text"})
    test = test.rename(columns={"message": "text"})

    # Labels must be of dtype int
    train["label"] = train["label"].astype(int)
    test["label"] = test["label"].astype(int)

    # Convert to Hugging Face Dataset format
    train = datasets.Dataset.from_pandas(train)
    test = datasets.Dataset.from_pandas(test)

    # Load tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    tokenize = lambda sample: tokenizer(sample["text"], truncation=True)
    train = train.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    classifier = transformers.AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    training_args = transformers.TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
    )

    # Data collator pads sequences dynamically
    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = transformers.Trainer(
        model=classifier,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate()

if __name__ == "__main__":
    main()