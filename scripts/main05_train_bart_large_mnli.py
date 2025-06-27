"""
Apply model facebook/bart-large-mnli to classify sms texts

The model bart-large-mnli is a pretrained, zero-shot classification model, fine-tuned on the Multi-Genre Natural
Language Inference dataset. It is supposed to be used without further training by providing expected labels ("ham" and
"spam") and then can be applied to arbitrary text messages to return how likely the text entails the label.

We therefore can expect this model to perform poorly in comparison to models fine-tuned on the dataset at hand.
"""

import pandas as pd
import sklearn.metrics
import transformers
import mlflow.pyfunc
import tqdm

import source.io

def main():
    path_data = source.io.path_root / "data/splits/2025-06-19"
    train, test = [pd.read_csv(path_data / f"{split}.csv", index_col=False) for split in ['train', 'test']]

    # Hugging Face expects a column named 'text'
    train = train.rename(columns={"message": "text"})
    test = test.rename(columns={"message": "text"})

    # Labels must be of dtype int
    train["label"] = train["label"].map({False: "ham", True: "spam"})
    test["label"] = test["label"].map({False: "ham", True: "spam"})

    classifier = transformers.pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    labels = ["spam", "ham"]

    # Call to get top predicted label
    predict_label = lambda text: classifier(text, labels)["labels"][0]

    with mlflow.start_run(run_name="bart-large-mnli"):
        mlflow.set_tag("model_type", "llm")
        mlflow.log_param("model", "facebook/bart-large-mnli")
        mlflow.log_param("classification_type", "zero-shot")
        mlflow.log_param("labels", labels)

        # Predict all test samples
        y_true = test["label"].tolist()
        y_pred = [predict_label(text) for text in tqdm.tqdm(test["text"], unit='messages', desc='Running predictions')]

        # Evaluate
        acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label="spam")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        # Save predictions to file
        path_output = source.io.path_root/"data/bart_large_mnli"
        path_output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({
            "message": test["text"],
            "label": y_true,
            "prediction": y_pred
        }).to_csv(path_output/"predictions.csv", index=False)
        mlflow.log_artifact(str(path_output/"predictions.csv"))

        print("Zero-shot run logged. Accuracy:", acc)

if __name__ == "__main__":
    main()