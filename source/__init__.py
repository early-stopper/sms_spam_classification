
import pathlib

import pandas as pd
import nltk
import mlflow

# Configure pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 300)

# Configure NLTK
nltk_resources = {
    'punkt': 'tokenizers/punkt',
    'punkt_tab': 'tokenizers/punkt_tab',
    'stopwords': 'corpora/stopwords',
}
def get_nltk_resources():
    for name, path in nltk_resources.items():
        try:
            nltk.data.find(path)
            print(f"✔ Path {path} is available")
        except LookupError:
            print(f"✘ Path {path} is missing. Downloading resources...")
            nltk.download(name)

# Configure MLflow
# This ensures that all scripts producing MLflow logs write to the same location, wherever they are executed
mlflow.set_experiment("sms_spam_classification")
# path_mlruns = pathlib.Path(__file__).parent.parent.resolve()/'mlruns'
# mlflow.set_tracking_uri(f"file:///{path_mlruns.as_posix()}")