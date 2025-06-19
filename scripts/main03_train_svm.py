
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.models.signature
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.svm
import sklearn.metrics

import source.io
import source.models

def main():
    path_data = source.io.path_root/"data/splits/2025-06-19"
    X_train, X_test = [pd.read_csv(path_data/f"{split}.csv", index_col=False) for split in ['train', 'test']]
    y_train, y_test = X_train.pop('label'), X_test.pop('label')

    with mlflow.start_run(run_name="svm"):
        mlflow.set_tag("model_type", "classical")

        # Set up model
        model = sklearn.pipeline.Pipeline([
            ('ensure_series', source.models.EnsureSeries(column='message')), #NOTE: workaround to fix MLflow/scikit-learn incompatibility, see docstring
            ('pre_processor', sklearn.feature_extraction.text.TfidfVectorizer()),
            ('classifier', sklearn.svm.LinearSVC())
        ])
        mlflow.log_param("classifier", "LinearSVC")

        # Training
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation
        metrics = sklearn.metrics.classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
        print(metrics)
        for label, score in metrics.items():
            if isinstance(score, dict):
                for metric, val in score.items():
                    mlflow.log_metric(f"{label.replace(' ', '_')}_{metric.replace('-', '_')}", val)
            else:
                mlflow.log_metric(label, score)

        # Store some (X,y) pairs as examples on what signature the model expects/returns
        input_example = X_test.iloc[:5]
        signature = mlflow.models.signature.infer_signature(input_example, y_pred[:5])

        # Save model
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example
        )

if __name__ == "__main__":
    main()