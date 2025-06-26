"""
Generate datasets for training and testing models

Read in raw data, apply pre-processing steps and define splits for training and testing. The results are saved as CSVs
to data/splits/
"""

import re

import numpy as np
import pandas as pd
import nltk
import sklearn.model_selection
import matplotlib.pyplot as plt

import source.io

def preprocess_text(text):
    # Ensure lowercase text
    text = text.lower()

    # Remove non-alphabetic characters (keep spaces)
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize
    tokens = nltk.tokenize.word_tokenize(text)

    # Remove stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def main():
    path_data = source.io.path_root/'data'

    data = pd.read_csv(path_data/'uci_spam_collection/SMSSpamCollection', sep='\t', header=None, names=['label', 'raw_message'])
    data['label'] = data['label']=='spam'
    print(f"Dataset contains {len(data)} samples with {100*data['label'].sum()/len(data):.2f}% prevalence.")

    # Clean messages
    data['message'] = data['raw_message'].apply(preprocess_text)

    # View some messages
    n_samples =20
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'
    for i, row in data.sample(n_samples, random_state=194316).iterrows():
        color = RED if row['label'] else GREEN
        print(f"\n{color}Message : {row['raw_message']}")
        print(f"Cleaned : {row['message']}{RESET}")

    # Drop empty samples
    # A few very short or nonsensical messages are left completely empty by the pre-processing. Since it is only 6 out
    # of 5566 messages, we decide to drop them. Possible alternatives:
    #  - restrict stop word removal to messages of minimum token length
    #  - replace empty strings with keyword such as "[empty]"
    is_empty = data['message'].str.strip()==''
    data = data[~is_empty]
    print(f"Dropped {is_empty.sum()} of {len(data)} samples where pre-processing left messages completely empty.")

    # Split train/test
    data['split'] = 'train'
    test_split = 0.2
    rng = np.random.default_rng()
    ind = rng.choice(data.index, int(test_split * len(data)))
    data.loc[ind, 'split'] = 'test'
    print(data['split'].value_counts())

    # Save splits
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    path_splits = path_data/'splits'/today
    path_splits.mkdir(parents=True, exist_ok=True)
    for split, ind in data.groupby('split').groups.items():
        data.loc[ind, ['message', 'label']].to_csv(path_splits/f"{split}.csv", index=False)
    print(f"Successfully saved splits to {path_splits}.")


if __name__ == "__main__":
    main()