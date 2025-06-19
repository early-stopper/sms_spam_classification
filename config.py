
import pandas as pd
import nltk

# Configure printing options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 1000)
pd.set_option('display.width', 300)

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