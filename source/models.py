
import pandas as pd
import sklearn.base


class EnsureSeries(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Ensures input is transformed to a pandas Series

    This is a workaround to get MLflow to work with sklearn pipelines expecting pd.Series:
    - Type checking in MLflow allows pd.DataFrames, but no pd.Series
    - Some scikit-learn models like sklearn.feature_extraction.text.TfidfVectorizer assume 1D input like pd.Series, but
      won't accept pd.DataFrames even if they only contain a single column.

    Therefore, this model is a simple preprocessing step that forwards inputs of type pd.Series as is, but extracts a
    certain column from pd.DataFrames.
    """


    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.column]
        elif isinstance(X, pd.Series):
            return X
        else:
            raise ValueError("Input must be a DataFrame or Series")