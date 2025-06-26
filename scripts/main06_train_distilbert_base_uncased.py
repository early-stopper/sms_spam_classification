
import numpy as np
import pandas as pd
import sklearn.metrics
import transformers
import datasets
import mlflow.transformers

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

    path_output = source.io.path_root/"data/distilbert_base_uncased"
    path_output.mkdir(parents=True, exist_ok=True)
    training_args = transformers.TrainingArguments(
        output_dir=path_output/"results",
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=path_output/"logs",
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
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # mlflow.set_tracking_uri("file:///C:/Users/user/projects/sms_spam_classification/mlruns")
    with mlflow.start_run(run_name="distilbert_base_uncased"):
        mlflow.set_tag("model_type", "llm")
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
        mlflow.log_param("batch_size", training_args.per_device_train_batch_size)

        # Train the model
        trainer.train()

        # Evaluate
        predictions = trainer.predict(test)
        mlflow.log_metrics(predictions.metrics)

        # Save predictions
        results = pd.DataFrame({
            'message': [x["text"] for x in test],
            'label': predictions.label_ids,
            'prediction': predictions.predictions.argmax(axis=1)
        })
        results['label'] = results['label'].map({0: "ham", 1: "spam"})
        results['prediction'] = results['prediction'].map({0: "ham", 1: "spam"})
        results.to_csv(path_output/"predictions.csv", index=False)

        # Log tokenizer and model
        # NOTE: MLflow cannot handle arbitrary huggingface models. Instead, we first must wrap tokenizer and classifier
        # in a pipeline
        pipeline = transformers.pipeline(
            task="text-classification",
            model=classifier,
            tokenizer=tokenizer,
            return_all_scores=False  # True to get full probability distribution
        )
        mlflow.transformers.log_model(
            transformers_model=pipeline,
            name="model",
            input_example="Congratulations! You’ve won a free ticket.",
            registered_model_name="distilbert_base_uncased"
        )

        # Log training args
        mlflow.log_dict(training_args.to_dict(), "training_args.json")

        print("Logged run to:", mlflow.get_artifact_uri())

if __name__ == "__main__":
    main()