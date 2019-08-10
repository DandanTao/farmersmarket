#!/usr/bin/python3
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
spacy_parser = spacy.load('en')

top_terms = 'top_terms_yelp.txt'

def top_20_terms_no_repeat(idxs, words, coef):
    added = set()
    final_word = []
    final_coef = []

    for idx in idxs:
        word = words[idx[0][0]]
        if word in added:
            continue
        added.add(word)
        final_word.append(word)
        final_coef.append(round(coef[idx[0][0]], 4))

        if len(added) == 20:
            break
    return (final_word, final_coef)

def print_top20_words(testname, vectorizer, pipe):
    features = vectorizer.get_feature_names()
    with open(top_terms, 'a') as f:
        f.write(f"RESULTS FOR {testname}\n")
        for i, coef in enumerate(pipe.steps[2][1].coef_):
            sorted_coef = sorted(coef)

            high_idx = [np.where(coef == d) for d in sorted_coef[::-1]]

            curr_class = pipe.steps[2][1].classes_[i]
            curr_coef = pipe.steps[2][1].coef_[i]

            top_10, top_coef = top_20_terms_no_repeat(high_idx, features, curr_coef)
            f.write(f"TOP 20 for {curr_class}\n\t{top_10}\n\t{top_coef}\n")

        f.write('-'*150)
        f.write('\n')

#tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
def spacy_tokenizer(sentence):
    tokens = spacy_parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens

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
        tokens = spacy_parser(sentence)
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
    pipe.fit(data.Sentences, data.Label)
    print_top20_words("SVC with BOG model", vectorizer, pipe)
    return pipe

def MultinomialNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 3), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    pipe = Pipeline([('vect', vectorizer),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', MultinomialNB())])

    pipe.fit(data.Sentences, data.Label)
    print_top20_words("MNB with TFIDF model", vectorizer, pipe)
    return pipe

def BernoulliNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 1), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    pipe = Pipeline([('vect', vectorizer),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', BernoulliNB())])

    pipe.fit(data.Sentences, data.Label)
    print_top20_words("BNB with TFIDF model", vectorizer, pipe)
    return pipe

def ComplementNBModel(data):
    """
    Uses Multinomial Naive Bayes Classifer to train
    """
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1,3), stop_words='english')

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    pipe = Pipeline([('vect', vectorizer),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', ComplementNB())])

    pipe.fit(data.Sentences, data.Label)
    # print_top20_words("CNB with TFIDF model", tfidf, pipe)
    return pipe

def SGDModel(data):
    """
    Uses Multinomial Stochastic Gradient Descent to train
    """
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=50, tol=1e-3)

    # apply CountVectorizer(), TfidfTransformer(), (transforms) and final estimator(classifier)
    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))
    pipe = Pipeline([('vect', vectorizer),
                    ('tfidf', TfidfTransformer()),
                    ('classifier', classifier)])

    pipe.fit(data.Sentences, data.Label)
    print_top20_words("SGD with TFIDF model", vectorizer, pipe)
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
    pipe.fit(data.Sentences, data.Label)
    print_top20_words("LSVC with BOW model", vectorizer, pipe)
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

    # tokenizer to parse sentence, removing stopwords, removing punctuations and generate tokens using Spacy used for CountVectorizer()
    def spacy_tokenizer(sentence):
        tokens = spacy_parser(sentence)
        tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
        tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
        return tokens

    vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, ngram_range=(1,1))

    classifier = LogisticRegression(random_state=0, multi_class="multinomial", solver="newton-cg")

    # apply predictors(), vectorizer (transforms) and final estimator(classifier)
    pipe = Pipeline([("cleaner", predictors()),
                     ('vectorizer', vectorizer),
                     ('classifier', classifier)])

    # Load sample data
    pipe.fit(data.Sentences, data.Label)
    print_top20_words("LR with BOW model", vectorizer, pipe)

    return pipe
