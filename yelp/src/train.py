import string
import spacy
from copy import deepcopy as DP

from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

punctuations = string.punctuation
parser = spacy.load('en')

#tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens

def remove_white_space(dataframe):
    for index, row in dataframe.iterrows():
        row['Label'] = row['Label'].rstrip().lstrip()
    return dataframe

def parse_csv_by_class(file):
    df = pd.read_csv(file)
    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']

    df_aval = df[df.Label.str.contains('availability')]
    df_environ = df[df.Label.str.contains('environment')]
    df_quality = df[df.Label.str.contains('quality')]
    df_safety = df[df.Label.str.contains('safety')]
    df_nonrel = df[df.Label == "Non-relevant"]

    return (df_aval, df_environ, df_quality, df_safety, df_nonrel)

def parse_csv_relevant_non_relevant(file):
    """
    reads csv file data with labels and comment
    Parse into 2 category: Relevant and Non-relevant
    """
    import pandas as pd
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")

    for index, row in df.iterrows():
        if row['Label'] != 'Non-relevant':
            row['Label'] = 'relevant'

    train, test = train_test_split(df, test_size=0.2)
    return train.to_numpy(), test.to_numpy()

def parse_csv_remove_multiclass(file):
    """
    reads csv file data with labels and comment
    Discards only multiclass objects
    """
    import pandas as pd
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("None")
    df = remove_white_space(df)
    # drop all sentence with multiple labels
    df=df[~df.Label.str.contains(",")]

    # drop all sentences with label 'covinience'
    # df = df.replace('covinience', 'convenience')
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']

    train, test = train_test_split(df, test_size=0.2)
    return train.to_numpy(), test.to_numpy()

def parse_csv_discard_non_relevant(file):
    """
    reads csv file data with labels and comment
    Discards all non-relevant sentences and multiclass sentences
    """
    import pandas as pd
    df = pd.read_csv(file)

    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("None")
    df=remove_white_space(df)
    # drop all sentence with multiple labels
    df=df[df.Label != 'None']
    df=df[~df.Label.str.contains(",")]

    # drop all sentences with label 'covinience'
    # df = df.replace('covinience', 'convenience')
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']

    train, test = train_test_split(df, test_size=0.2)
    return train.to_numpy(), test.to_numpy()

def SVCModel(data):
    """
    Uses Multinomial Support Vector Machine to train
    """
    #transformers using Spacy
    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}
        # cleans the text
        def clean_text(self, text):
            return text.strip().lower()

    #tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
    def spacy_tokenizer(sentence):
        tokens = parser(sentence)
        tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
        tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
        return tokens

    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))

    classifier = SVC(gamma='auto')

    # apply predictors(), vectorizer (transforms) and final estimator(classifier)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    pipe.fit([x[0] for x in data], [x[1] for x in data])

    return pipe

def MultinomialNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 3), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,3))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())])

    # f = tfidf.fit_transform(np.array(features)).toarray()
    # mnb = snb.MultinomialNB()
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(features)
    # X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe
def BernoulliNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 1), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', BernoulliNB())])

    # f = tfidf.fit_transform(np.array(features)).toarray()
    # mnb = snb.MultinomialNB()
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(features)
    # X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe

def ComplementNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1,3), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', ComplementNB())])

    # f = tfidf.fit_transform(np.array(features)).toarray()
    # mnb = snb.MultinomialNB()
    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(features)
    # X_train_tfidf = TfidfTransformer().fit_transform(X_train_counts)

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe

def SGDModel(data):
    """
    Uses Multinomial Stochastic Gradient Descent to train
    """
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3)

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    pipe = Pipeline([('vect', CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', classifier)])

    pipe.fit([x[0] for x in data], [x[1] for x in data])
    return pipe

def LSVCModel(data):
    """
    Uses Multinomial Support Vector Machine to train
    """
    #transformers using Spacy
    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}
        # cleans the text
        def clean_text(self, text):
            return text.strip().lower()

    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    classifier = LinearSVC()

    # apply predictors(), vectorizer (transforms) and final estimator(classifier)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    pipe.fit([x[0] for x in data], [x[1] for x in data])

    return pipe

def LogisticRegressionModel(data):
    """
    Uses Multinomial Support Vector Machine to train
    """
    #transformers using Spacy
    class predictors(TransformerMixin):
        def transform(self, X, **transform_params):
            return [self.clean_text(text) for text in X]
        def fit(self, X, y=None, **fit_params):
            return self
        def get_params(self, deep=True):
            return {}
        # cleans the text
        def clean_text(self, text):
            return text.strip().lower()

    #tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
    def spacy_tokenizer(sentence):
        tokens = parser(sentence)
        tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
        tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
        return tokens

    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    classifier = LogisticRegression(random_state=0)

    # apply predictors(), vectorizer (transforms) and final estimator(classifier)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    pipe.fit([x[0] for x in data], [x[1] for x in data])

    return pipe
