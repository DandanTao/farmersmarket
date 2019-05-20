import pandas as pd
import numpy as np
from models import SGDModel, LogisticRegressionModel
from csv_parser import parse_csv_by_class
from RB_classification import getTFIDF, getCWScore
from preprocess import preprocess
import time
from pprint import pprint
import os

TRAIN_FILE_PATH = "../data/yelp_labelling_1000.csv"
TEST_FILE_PATH = "../data/separate_sentence.txt"



def testSGD10000(train_file, test_file):
    """
    train_file: csv file containing sentences and label
    test_file: plain text file with newline separated sentences
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class(train_file)
    frames = [df_aval, df_environ, df_quality, df_safety, df_nonrel]
    df_ML = pd.concat(frames)

    test_sen = []
    with open(test_file) as f:
        for i, line in enumerate(f):
            test_sen.append(line.rstrip().lstrip())
            
    #==================================SGD=====================================#
    SGD_start = time.time()
    SGDPipe = SGDModel(df_ML.to_numpy())
    SGD_Pred = SGDPipe.predict(test_sen)
    SGD_end = time.time()

    #===================================LR=====================================#
    LR_start = time.time()
    LRPipe = LogisticRegressionModel(df_ML.to_numpy())
    LR_Pred = LRPipe.predict(test_sen)
    LR_end = time.time()

    #=================================TFIDF====================================#
    TFIDF_start = time.time()
    aval_TFIDF = getTFIDF(df_aval)
    environ_TFIDF = getTFIDF(df_environ)
    quality_TFIDF = getTFIDF(df_quality)
    safety_TFIDF = getTFIDF(df_safety)
    nonrel_TFIDF = getTFIDF(df_nonrel)
    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'Non-relevant'}

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
            predLabel = 'Non-relevant'
        else:
            predLabel = idx_to_Label[all_sc.index(max_sc)]
        tfidf_pred.append(predLabel)
    TFIDF_end = time.time()

    print(f"TIME IT TOOK TO TRAIN {len(test_sen)} SENTENCES(sec): \
    \n\tSGD: {SGD_end-SGD_start}\n\tLR: {LR_end-LR_start}\n\tTFIDF: {TFIDF_end-TFIDF_start}")

    res = {"Sentence":test_sen, "SGD_Pred":SGD_Pred, "LR_Pred":LR_Pred, "TFIDF_Pred":tfidf_pred}
    df = pd.DataFrame(data=res)
    os.chdir("../data")
    df.to_excel("all_reviews.xlsx", index=False)

testSGD10000(TRAIN_FILE_PATH, TEST_FILE_PATH)
