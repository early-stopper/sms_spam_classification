
import pandas as pd
import numpy as np
import sklearn.metrics

import source.io

def get_color(value: float) -> str:
    """Get cell color based on value

    Value <=0.5            ==> color='red'
    Value in (0.5, 0.75)   ==> color between 'red' and 'yellow'
    Value in (0.75, 1.0)   ==> color between 'yellow' and 'green'
    """
    
    value = np.clip(value, 0.0, 1.0)
    if value <= 0.5:
        r, g, b = 255, 0, 0
    elif value <= 0.75:
        # red to yellow
        r, g, b = 255, int(255 * (value-0.5) * 4), 0
    else:
        # yellow to green
        r, g, b = int(255 * (2 - 4 * (value-0.5))), 255, 0
    return f"\033[38;2;{r};{g};{b}m"  # Text color

def get_format(value: float, is_best: bool, width=9) -> str:
    color = get_color(value)
    highlighted = "\033[1;3m" if is_best else ""
    reset = "\033[0m"
    return f"{color}{highlighted}{value:>{width}.3f}{reset}"

def main():
    models = [
        "logistic_regression",
        "support_vector_machine",
        "naive_bayes",
        "bart_large_mnli",
        "distilbert_base_uncased"
    ]

    metrics = pd.DataFrame(
        index=models,
        columns = ["accuracy", "recall", "precision", "f1_score"]
    )
    for model in models:
        predictions = pd.read_csv(source.io.path_root/"data"/model/"predictions.csv")
        y_true = predictions["label"].map({"ham": 0, "spam": 1})
        y_pred = predictions["prediction"].map({"ham": 0, "spam": 1})

        metrics.loc[model, "accuracy"] = sklearn.metrics.accuracy_score(y_true, y_pred)
        metrics.loc[model, "recall"] = sklearn.metrics.recall_score(y_true, y_pred)
        metrics.loc[model, "precision"] = sklearn.metrics.precision_score(y_true, y_pred)
        metrics.loc[model, "f1_score"] = sklearn.metrics.f1_score(y_true, y_pred)

    # Print colored metrics
    col_width = 12
    header = f"{'Model':<30} " + " ".join(f"{col:>{col_width}}" for col in metrics.columns)
    print(header)
    print("-" * len(header))
    best = metrics == metrics.max() # betst value for each column
    for index, row in metrics.iterrows():
        line = f"{index:<30} "
        for col in metrics.columns:
            value = row[col]
            line += f"{get_format(value, best.loc[index, col], width=col_width)} "
        print(line)


if __name__ == "__main__":
    main()