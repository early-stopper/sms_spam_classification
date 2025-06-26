# sms_spam_classification
Running experiments on classifying spam messages in the "UCI SMS Spam Collection" dataset

## Data

The UCI spam collection can be downloaded [here](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) as a ZIP
folder. Unpacking it to the `data/uci_spam_collection/` subdirectory should reveal 2 files: the `readme` file with
detailed information on the dataset and the `SMSSpamCollection` containing the actual data as tab-separated, headerless 
CSV with the label `spam` or `ham` in the first column and the corresponding SMS text in the second column.

Run `scripts/main01_pre_process_splits.py` to read in the dataset, run cleaning steps on the text messages and to
generate splits for traininhg and testing models. The splits are written to `data/splits`.

## Models

Our goal is to compare modern large language models with classical approaches from natural language processing. To this
end, we chose the following candidates for classifiers to distinct spam from ham messages.

### Classical

TODO: explain TfidfVectorizer

#### Logistic Regression

#### Supporting Vector Machine

#### Naïve Bayes

### Large Language Models

#### Distilbert Base Uncased

#### Bart Large MNLI

## Evaluations

We decided to work with [MLflow](https://mlflow.org/) to ease evaluation of the models. As long as training and testing 
pipelines are adjusted appropriately to provide hooks for parameters and models to be tracked, progress can be
monitored by running the MLflow browser UI.

To open up the MLflow UI, first make sure that the virtual environment is activated by running the command

```.\.venv\Scripts\activate```

on a command prompt located at the root directory of this project. Then change to directory `.\scripts` and run

```mlflow ui```

to initialise the MLflow server. Follow the instructions to open up the UI in the browser.
