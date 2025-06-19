
import pandas as pd
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.metrics

import source.io

# TODO: consider tuning parameters of TfidfVectorizer and MultinomialNB

def main():
    path_data = source.io.path_root/"data/splits/2025-06-19"
    train, test = [pd.read_csv(path_data/f"{split}.csv", index_col=False) for split in ['train', 'test']]
    X_train, X_test = train['message'], test['message']
    y_train, y_test = train['label'], test['label']

    model = sklearn.pipeline.Pipeline([
        ('pre_processor', sklearn.feature_extraction.text.TfidfVectorizer()),
        ('classifier', sklearn.naive_bayes.MultinomialNB())
    ])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = sklearn.metrics.classification_report(y_test, y_pred, target_names=["ham", "spam"])
    print(metrics)


if __name__ == "__main__":
    main()