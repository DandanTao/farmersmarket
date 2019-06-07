#!/usr/bin/python3
from models import *
from csv_parser import *
from preprocess import preprocess, preprocessTFIDF
import numpy as np
import gensim
from gensim import corpora, models

from ML_analyze import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

from RB_analyze import RB_bar_graph

import LDA

import matplotlib.pyplot as plt

PATH1="../data/yelp_labelling_1000.csv"
PATH2="../data/1000_more_yelp.csv"
PATH3="../data/2000_yelp_labeled.csv"

def wordCount(df, preprocessor):
    """
    Preprocesses document and returns a dictinary of
    {word:word_count | for all word in preprocessed document}
    """
    res = {}
    count = 0
    for row in df.iterrows():
        for word in preprocessor(row[1]["Sentences"]).split(" "):
            if word in res:
                res[word] += 1
            else:
                res[word] = 1
            count += 1
    return (res, count)

def sortByValue(dict):
    """
    Returns a list of tuple that is sorted by value
    """
    return sorted(wordCount(dict, preprocess)[0].items(), key=lambda x: x[1], reverse=True)

def wordCountRuleBase(file_path):
    """
    BOG model
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(file_path)

    aval = sortByValue(df_aval)
    environ = sortByValue(df_environ)
    qual =  sortByValue(df_quality)
    safety = sortByValue(df_safety)
    nonrel = sortByValue(df_nonrel)

    return (aval, environ, qual, safety, nonrel)

def TFNormalize(wordcount, num_terms):
    """
    Preprocesses document and returns a dictinary of
    {word:word_count/total_word_count | for all word in preprocessed document}
    """
    res = {}
    total = 0

    for i, (k, v) in enumerate(wordcount[:num_terms]):
        total += v

    for i in range(0, num_terms):
        res[wordcount[i][0]] = wordcount[i][1] / total
    return res

def getCWScore(TFDict, sen, preprocessor):
    """
    Given a sentence(@param sen), it computes the score of the preprocessed
    sen using the TF score.
    """
    sum = 0
    for word in preprocessor(sen).split(" "):
        if word in TFDict:
            sum += TFDict[word]
    return sum

def runWordCountRuleBase(file_path, test, num_terms=20):
    """
    Classic rule based classification only using term frequency
    """
    aval, environ, qual, safety, nonrel = wordCountRuleBase(file_path)
    aval_dict = TFNormalize(aval, num_terms)
    environ_dict = TFNormalize(environ, num_terms)
    qual_dict = TFNormalize(qual, num_terms)
    safety_dict = TFNormalize(safety, num_terms)
    nonrel_dict = TFNormalize(nonrel, num_terms)

    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'non-relevant'}
    count = 0
    correct = 0
    true_label = []
    pred_label = []
    for idx, row in test.iterrows():
        aval_sc = getCWScore(aval_dict, row['Sentences'], preprocess)
        environ_sc = getCWScore(environ_dict, row['Sentences'], preprocess)
        qual_sc = getCWScore(qual_dict, row['Sentences'], preprocess)
        safety_sc = getCWScore(safety_dict, row['Sentences'], preprocess)
        nonrel_sc = getCWScore(nonrel_dict, row['Sentences'], preprocess)
        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            continue

        trueLabel = row["Label"]
        predLabel = idx_to_Label[all_sc.index(max_sc)]
        pred_label.append(predLabel)
        true_label.append(trueLabel)
    accuracy = accuracy_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))
    recall = recall_score(true_label, pred_label, average='weighted')
    precision = precision_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))

    metrics = [accuracy, f1, recall, precision]
    print(f"BOW\t[accuracy: {accuracy}\tF1: {f1}\tRecall: {recall}\tPrecision: {precision}]")
    return np.array(metrics), true_label, pred_label

def computeTF(df):
    """
    Computes term frequency in docs
    """
    dict, count = wordCount(df, preprocessTFIDF)
    for k, v in dict.items():
        dict[k] /= count
    return dict

def computeIDF(df):
    """
    Computes inverse document frequency in docs
    """
    N = len(df)
    words = set()
    all_sen = []
    for i, row in df.iterrows():
        sen = preprocessTFIDF(row["Sentences"])
        all_sen.append(sen)
        for word in sen.split(" "):
            words.add(word)

    res = {}
    for word in words:
        n = 1 #smoothing
        for sen in all_sen:
            if word in sen:
                n += 1
        res[word] = np.log(N / n)

    return res

def computeTFIDF(TF, IDF):
    """
    Computes the TFIDF of a docuemnt for each word
    """
    res = {}
    for k, v in TF.items():
        res[k] = v * IDF[k]
    return res

def getTFIDF(df):
    """
    Returns a dictionary of {word:tfidf_score | for all word in document}
    """
    TF = computeTF(df)
    IDF = computeIDF(df)
    return computeTFIDF(TF, IDF)

def runTFIDFRuleBase(file_path, test):
    """
    Deterministic rule based classification using TFIDF
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(file_path)
    aval_TFIDF = getTFIDF(df_aval)
    environ_TFIDF = getTFIDF(df_environ)
    quality_TFIDF = getTFIDF(df_quality)
    safety_TFIDF = getTFIDF(df_safety)
    nonrel_TFIDF = getTFIDF(df_nonrel)

    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'non-relevant'}
    count = 0
    correct = 0
    true_label = []
    pred_label = []
    for idx, row in test.iterrows():
        aval_sc = getCWScore(aval_TFIDF, row['Sentences'], preprocess)
        environ_sc = getCWScore(environ_TFIDF, row['Sentences'], preprocess)
        qual_sc = getCWScore(quality_TFIDF, row['Sentences'], preprocess)
        safety_sc = getCWScore(safety_TFIDF, row['Sentences'], preprocess)
        nonrel_sc = getCWScore(nonrel_TFIDF, row['Sentences'], preprocess)
        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            predLabel = 'non-relevant'
        else:
            predLabel = idx_to_Label[all_sc.index(max_sc)]
        trueLabel = row["Label"]

        pred_label.append(predLabel)
        true_label.append(trueLabel)

    accuracy = accuracy_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))
    recall = recall_score(true_label, pred_label, average='weighted')
    precision = precision_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))

    metrics = [accuracy, f1, recall, precision]
    print(f"TFIDF\t[accuracy: {accuracy}\tF1: {f1}\tRecall: {recall}\tPrecision: {precision}]")
    return np.array(metrics), true_label, pred_label

def getLDATFIDFModel(df, filter=True):
    """
    LDA Topic Modeling model using TFIDF
    """
    processed_docs = df["Sentences"].map(LDA.preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if filter:
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    return gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=2, workers=4), dictionary

def getLDABOWModel(df, filter=True):
    """
    LDA Topic Modeling model using Bag Of Words
    """
    processed_docs = df["Sentences"].map(LDA.preprocess)
    dictionary = gensim.corpora.Dictionary(processed_docs)
    if filter:
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
    return gensim.models.LdaMulticore(bow_corpus, num_topics=10, id2word=dictionary, passes=2, workers=2), dictionary

def runLDARuleBase(file_path, test):
    """
    Text classification using LDA topic modeling
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(file_path)

    aval_lda_model, aval_dictionary = getLDABOWModel(df_aval, filter=True)
    environ_lda_model, environ_dictionary = getLDABOWModel(df_environ, filter=True)
    quality_lda_model, quality_dictionary = getLDABOWModel(df_quality, filter=True)
    safety_lda_model, safety_dictionary = getLDABOWModel(df_safety, filter=False)
    nonrel_lda_model, nonrel_dictionary = getLDABOWModel(df_nonrel, filter=True)

    aval_lda_model_tfidf, _ = getLDATFIDFModel(df_aval, filter=True)
    environ_lda_model_tfidf, _ = getLDATFIDFModel(df_environ, filter=True)
    quality_lda_model_tfidf, _ = getLDATFIDFModel(df_quality, filter=True)
    safety_lda_model_tfidf, _ = getLDATFIDFModel(df_safety, filter=False)
    nonrel_lda_model_tfidf, _ = getLDATFIDFModel(df_nonrel, filter=True)

    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'non-relevant'}
    count = 0
    bow_correct = 0
    tfidf_correct = 0
    for idx, row in test.iterrows():
        trueLabel = row["Label"]
        text = row['Sentences']
        scores = []
        scores.append(LDA.score_doc(text, aval_dictionary, aval_lda_model, aval_lda_model_tfidf))
        scores.append(LDA.score_doc(text, environ_dictionary, environ_lda_model, environ_lda_model_tfidf))
        scores.append(LDA.score_doc(text, quality_dictionary, quality_lda_model, quality_lda_model_tfidf))
        scores.append(LDA.score_doc(text, safety_dictionary, safety_lda_model, safety_lda_model_tfidf))
        scores.append(LDA.score_doc(text, nonrel_dictionary, nonrel_lda_model, nonrel_lda_model_tfidf))

        bow_sc = [x[0] for x in scores]
        tfidf_sc = [x[1] for x in scores]

        count += 1
        if idx_to_Label[bow_sc.index(max(bow_sc))] == trueLabel:
            bow_correct += 1

        if idx_to_Label[tfidf_sc.index(max(tfidf_sc))] == trueLabel:
            tfidf_correct += 1

    print(f"LDA TOPIC MODELING WITH BOW: {bow_correct / count}")
    print(f"LDA TOPIC MODELING WITH TFIDF: {tfidf_correct / count}")

    return (bow_correct / count, tfidf_correct / count)

def merge_list(m1, m2):
    """
    merge two list
    """
    for x in m2:
        m1.append(x)
    return m1

def run_all(iter, file_path, test, confusion_matrix=False, bar_graph=False):
    bow = []
    tfidf = []
    bow_lda = []
    tfidf_lda = []

    all_bow_true = []
    all_bow_pred = []

    all_tfidf_true = []
    all_tfidf_pred = []

    acc_bow = np.zeros(4)
    acc_tfidf = np.zeros(4)
    for i in range(iter):
        print(f"EPOCH {i+1}")

        bow_metrics, bow_true, bow_pred = runWordCountRuleBase(file_path, test, num_terms=100)
        tfidf_metrics, tfidf_true, tfidf_pred = runTFIDFRuleBase(file_path, test)

        all_bow_true = merge_list(all_bow_true, bow_true)
        all_bow_pred = merge_list(all_bow_pred, bow_pred)

        all_tfidf_true = merge_list(all_tfidf_true, tfidf_true)
        all_tfidf_pred = merge_list(all_tfidf_pred, tfidf_pred)
        # runLDARuleBase(PATH2, test)

        acc_bow += bow_metrics
        acc_tfidf += tfidf_metrics
        print()
    if confusion_matrix:
        plot_confusion_matrix(all_bow_true, all_bow_pred,
                                  normalize=True,
                                  title="BOW Confusion Matrix",
                                  cmap=plt.cm.Blues)

        plt.show()
        # print(tfidf_true)
        # print(tfidf_pred)
        plot_confusion_matrix(all_tfidf_true, all_tfidf_pred,
                                  normalize=True,
                                  title="TFIDF Confusion Matrix",
                                  cmap=plt.cm.Blues)
        plt.show()
    acc_bow /= iter
    acc_tfidf /= iter
    if bar_graph:
        RB_bar_graph(acc_bow, acc_tfidf)
    print(f"Bow Overall Stat[Accuracy, F1, Recall, Precision]: {list(acc_bow)}")
    print(f"TFIDF Overall Stat[Accuracy, F1, Recall, Precision]: {list(acc_tfidf)}")

    return (acc_bow, acc_tfidf, all_bow_pred, all_tfidf_pred)

if __name__ == '__main__':
    import sys
    iter = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    test = random_test_data_v1(PATH3, size=0.4)
    run_all(iter, PATH3, test, confusion_matrix=True, bar_graph=True)
