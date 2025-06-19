
import pandas as pd
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.metrics

import source.io

def main():
    path_data = source.io.path_root/"data/splits/2025-06-19"
    train, test = [pd.read_csv(path_data/f"{split}.csv", index_col=False) for split in ['train', 'test']]
    X_train, X_test = train['message'], test['message']
    y_train, y_test = train['label'], test['label']

    model = sklearn.pipeline.Pipeline([
        ('pre_processor', sklearn.feature_extraction.text.TfidfVectorizer()),
        ('classifier', sklearn.linear_model.LogisticRegression(max_iter=1000))
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = sklearn.metrics.classification_report(y_test, y_pred, target_names=["ham", "spam"])
    print(metrics)


if __name__ == "__main__":
    main()