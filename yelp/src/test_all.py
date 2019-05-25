import pandas as pd
import numpy as np
from models import SGDModel, LogisticRegressionModel
from csv_parser import parse_csv_by_class_v1, parse_csv_by_class_v0
from RB_classification import getTFIDF, getCWScore
from preprocess import preprocess
import time
from pprint import pprint
import os
import statistics
TRAIN_FILE_PATH1 = "../data/yelp_labelling_1000.csv"
TRAIN_FILE_PATH2 = "../data/1000_more_yelp.csv"
TRAIN_FILE_PATH3 = "../data/2000_yelp_labeled.csv"
TEST_FILE_PATH = "../data/separate_sentence.txt"

def common(l):
    return statistics.mode(l)

def get_vote_pred(l1, l2, l3):
    vote_pred = []
    for x, y, z in zip(l1, l2, l3):
        l = [x, y, z]
        try:
            most_common = common(l)
        except:
            most_common = "N/A"
        vote_pred.append(most_common)
    return vote_pred

def test_all_methods(train_file, test_file, parser=parse_csv_by_class_v1):
    """
    train_file: csv file containing sentences and label
    test_file: plain text file with newline separated sentences
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel \
                    = parser(train_file)
    frames = [df_aval, df_environ, df_quality, df_safety, df_nonrel]
    df_ML = pd.concat(frames)

    test_sen = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            test_sen.append(line.rstrip().lstrip())

    #==================================SGD=====================================#
    SGD_start = time.time()
    SGDPipe = SGDModel(df_ML)
    SGD_Pred = SGDPipe.predict(test_sen)
    SGD_end = time.time()

    #===================================LR=====================================#
    LR_start = time.time()
    LRPipe = LogisticRegressionModel(df_ML)
    LR_Pred = LRPipe.predict(test_sen)
    LR_end = time.time()

    #=================================TFIDF====================================#
    TFIDF_start = time.time()
    aval_TFIDF = getTFIDF(df_aval)
    environ_TFIDF = getTFIDF(df_environ)
    quality_TFIDF = getTFIDF(df_quality)
    safety_TFIDF = getTFIDF(df_safety)
    nonrel_TFIDF = getTFIDF(df_nonrel)
    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'non-relevant'}

    tfidf_pred = []
    for sen in test_sen:
        aval_sc = getCWScore(aval_TFIDF, sen, preprocess)
        environ_sc = getCWScore(environ_TFIDF, sen, preprocess)
        qual_sc = getCWScore(quality_TFIDF, sen, preprocess)
        safety_sc = getCWScore(safety_TFIDF, sen, preprocess)
        nonrel_sc = getCWScore(nonrel_TFIDF, sen, preprocess)
        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            predLabel = 'non-relevant'
        else:
            predLabel = idx_to_Label[all_sc.index(max_sc)]
        tfidf_pred.append(predLabel)
    TFIDF_end = time.time()

    print(f"TIME IT TOOK TO TRAIN {len(test_sen)} SENTENCES(sec): \
    \n\tSGD: {SGD_end-SGD_start}\n\tLR: {LR_end-LR_start}\n\tTFIDF: {TFIDF_end-TFIDF_start}")
    vote_pred = get_vote_pred(SGD_Pred, LR_Pred, tfidf_pred)
    # print(vote_pred)
    res = {"Sentences":test_sen, "SGD_Pred":SGD_Pred, "LR_Pred":LR_Pred, "TFIDF_Pred":tfidf_pred, "Vote_Pred":vote_pred}
    df = pd.DataFrame(data=res)
    # print(df)
    df.to_csv("../data/all_reviews_v2.csv", index=False)
if __name__ == '__main__':
    test_all_methods(TRAIN_FILE_PATH3, TEST_FILE_PATH, parser=parse_csv_by_class_v0)
