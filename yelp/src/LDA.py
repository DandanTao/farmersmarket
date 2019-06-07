#!/usr/bin/python3
"""
Refered code from
https://github.com/susanli2016/NLP-with-Python/blob/master/LDA_news_headlines.ipynb
"""
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
import nltk
import pandas as pd
from pprint import pprint
from gensim import corpora, models

def lemmatize_stemming(text):
    return SnowballStemmer('english').stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def score_doc(text, dictionary, lda_bow, lda_tfidf):
    bow_vec = dictionary.doc2bow(preprocess(text))
    bow_sc = None
    tfidf_sc = None
    for index, score in sorted(lda_bow[bow_vec], key=lambda tup: -1*tup[1]):
        #print("Score: {}\t Topic: {}".format(score, lda_bow.print_topic(index, 5)))
        bow_sc = score
        break

    for index, score in sorted(lda_tfidf[bow_vec], key=lambda tup: -1*tup[1]):
        #print("Score: {}\t Topic: {}".format(score, lda_tfidf.print_topic(index, 5)))
        tfidf_sc = score
        break
    return (bow_sc, tfidf_sc)

# PATH="../data/yelp_labelling_1000.csv"
# df=pd.read_csv(PATH)
# df = df.fillna("None")
# df = df[df['Label'] != "safety?"]
# df.drop(["Index", "Unnamed: 5"], axis=1, inplace=True)
# text = df[df["Label"] == "None"]
#
# processed_docs = text['Sentences'].map(preprocess)
# dictionary = gensim.corpora.Dictionary(processed_docs)
# dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
# bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
#
# tfidf = models.TfidfModel(bow_corpus)
# corpus_tfidf = tfidf[bow_corpus]
#
# lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2)
#
# # for idx, topic in lda_model.print_topics(-1):
# #     print('Topic: {} \nWords: {}'.format(idx, topic))
#
# lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4)
# # for idx, topic in lda_model_tfidf.print_topics(-1):
# #     print('Topic: {} Word: {}'.format(idx, topic))
#
# bow_stat=[]
# tfidf_stat=[]
# corpus_len=len(bow_corpus)
# for bow in bow_corpus:
#     for index, score in sorted(lda_model[bow], key=lambda tup: -1*tup[1]):
#         bow_stat.append(score)
#         #print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))
#         break
#
#     for index, score in sorted(lda_model_tfidf[bow], key=lambda tup: -1*tup[1]):
#         tfidf_stat.append(score)
#         #print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))
#         break
#
# bow_stat.sort()
# tfidf_stat.sort()
#
# print(f"Median Score using BOG model {bow_stat[corpus_len // 2]}")
# print(f"Median Score using TFIDF model{tfidf_stat[corpus_len // 2]}")
#
# print(f"Average Score using BOG model {sum(bow_stat)/corpus_len}")
# print(f"Average Score using TFIDF model{sum(tfidf_stat)/corpus_len}")
#
# text="Before you know it, your allergies will be gone."
# score_doc(text, dictionary, lda_model, lda_model_tfidf)
