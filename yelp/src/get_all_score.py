import json
import os
import numpy as np
import numpy.linalg as la
from pprint import pprint
import subprocess
import pandas as pd
import numpy as np
from models import SGDModel, LogisticRegressionModel, LSVCModel
from csv_parser import parse_csv_by_class_v1, parse_csv_by_class_v0
from RB_classification import getTFIDF, getCWScore
from preprocess import preprocess, clean_text
import time
from pprint import pprint
import os
import statistics
from test_all import get_vote_pred, common
from sentiment_anaylisis_ML_v2 import parse_excel_by_class

from merge_all_reviews import SEPARATE_SENTENCE_DIR, REVIEW_DIR

AVAIL = "availability"
ENVIRON = 'environment'
QUAL = 'quality'
SAFETY = 'safety'
NON_REL = 'non-relevant'

FM_INFO_DIR = '/Users/jaewooklee/farmers_market/yelp/data/all_FM_info.json'

TRAIN_FILE_PATH = "../data/2000_yelp_labeled.csv"
SEPARATE_SENTENCE_DIR = '/Users/jaewooklee/farmers_market/sentenceboundary/out/'

FM_REV_PER_ASP_DIR = '/Users/jaewooklee/farmers_market/yelp/src/reviews_per_fm_per_aspect/'
class FM_info:
    def __init__(self, dict):
        self.name = dict['name']
        self.alias = dict['alias']
        self.lat = dict['coordinates']['latitude']
        self.lng = dict['coordinates']['longitude']
        self.num_reviews = dict['review_count']
        self.overall = dict['rating']

    def __repr__(self):
        return f"{self.name} {self.lat} {self.lng}"

# class FM_reviews:
#     def __init__(self, fm_info, )

def apply_leftover(final_score, leftover, all_reviews):
    print("LEFTOVER ALGO")
    print(final_score)
    print(leftover)
    print(all_reviews)
    has_reviews = [0, 0, 0, 0]
    possitivness = []
    for i in range(0, len(all_reviews)):
        if sum(all_reviews[i]) == 0 or final_score[i] == None or final_score[i] == 5.0:
            pass
        else:
            possitivness.append(all_reviews[i][0] / sum(all_reviews[i]))
            has_reviews[i] = 1

    if len(possitivness) == 1:
        scores = [leftover]
    else:
        possitivness /= la.norm(possitivness, 1)
        possitivness -= (1/len(possitivness))

        scores = leftover * (1+possitivness)

    idx = 0
    for i, sc in enumerate(final_score):
        if sc is not None:
            if sc < 5.0:
                print(final_score[i])
                print(scores[idx])
                final_score[i] += scores[idx]
                idx += 1

    leftover = 0
    for i in range(0, len(final_score)):
        if final_score[i] != None and final_score[i] > 5.0:
            leftover += (final_score[i] - 5.0)
            final_score[i] = 5.0
    return (final_score, leftover)

def scoreing_algorithm(overall, all_reviews):
    print("SCORING ALGO")
    print(overall)
    print(all_reviews)
    all_zero = [sum(x) for x in all_reviews]
    if sum(all_zero) == 0:
        return [None, None, None, None]
    has_reviews = [0, 0, 0, 0]
    possitivness = []
    for i in range(0, len(all_reviews)):
        if sum(all_reviews[i]) == 0:
            pass
        else:
            possitivness.append(all_reviews[i][0] / sum(all_reviews[i]))
            has_reviews[i] = 1
    if len(possitivness) > 0 and sum(possitivness) > 0:
        possitivness /= la.norm(possitivness, 1)
        possitivness -= (1/len(possitivness))

        scores = overall * (1+possitivness)
    else:
        scores = [overall] * len(possitivness)

    leftover = 0
    for i in range(0, len(scores)):
        if scores[i] > 5.0:
            leftover += (scores[i] - 5.0)
            scores[i] = 5.0

    final_score = []
    i = 0
    for rev in has_reviews:
        if rev == 1:
            final_score.append(scores[i])
            i += 1
        else:
            final_score.append(None)

    while leftover > 1:
        final_score, leftover = apply_leftover(final_score, leftover, all_reviews)

    print("DONE SCORING {}".format(final_score))
    return final_score

def getLatLngFarmersMarket(file):
    aliasToLoc = {}
    data = None
    with open(file) as f:
        data = json.load(f)

    for fm in data['businesses']:
        aliasToLoc[fm['alias']] = FM_info(fm)

    return aliasToLoc

def TFIDF_predict(sentences, aval, environ, qual, safety, nonrel):
    idx_to_Label = {0:AVAIL, 1:ENVIRON, 2:QUAL, 3:SAFETY, 4:NON_REL}
    tfidf_pred = []

    for sen in sentences:
        aval_sc = getCWScore(aval, sen, preprocess)
        environ_sc = getCWScore(environ, sen, preprocess)
        qual_sc = getCWScore(qual, sen, preprocess)
        safety_sc = getCWScore(safety, sen, preprocess)
        nonrel_sc = getCWScore(nonrel, sen, preprocess)

        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            tfidf_pred.append(NON_REL)
        else:
            tfidf_pred.append(idx_to_Label[all_sc.index(max_sc)])

    return tfidf_pred

def convert_info_to_dict(fm_info, review_scores):
    return {"Name": fm_info.name, "Long": fm_info.lng, "Lat": fm_info.lat,
    "num_reviews": fm_info.num_reviews,"Overall":fm_info.overall,"Availability":review_scores[0],"Environment":review_scores[1],
    "Quality": review_scores[2], "Safety": review_scores[3], 'Alias': fm_info.alias}

def get_overall_score(alias):
    with open(REVIEW_DIR + alias + ".json") as f:
        data = json.load(f)
    return (data['aggregateRating']['reviewCount'], data['aggregateRating']['ratingValue'])

def count_possitive_reviews(reviews, label, pos_TFIDF, neg_TFIDF):
    positive = 0
    negative = 0

    for sen in reviews:
        pos_sc = getCWScore(pos_TFIDF, sen, preprocess)
        neg_sc = getCWScore(neg_TFIDF, sen, preprocess)

        all_sc = [neg_sc, pos_sc]
        max_sc = max(all_sc)
        if max_sc == 0 or all_sc.index(max_sc) == 1:
            positive += 1
        else:
            negative += 1

    return (positive, negative)

def write_rev_per_asp(alias, aspect, revs):
    with open(FM_REV_PER_ASP_DIR + alias + '_' + aspect + '.txt', 'w') as f:
        for rev in revs:
            f.write(rev)
            f.write('\n')

def get_all_score():
    df_aval, df_environ, df_quality, df_safety, df_nonrel \
                    = parse_csv_by_class_v1(TRAIN_FILE_PATH)
    frames = [df_aval, df_environ, df_quality, df_safety, df_nonrel]
    df_ML = pd.concat(frames)

    SGDPipe = SGDModel(df_ML)
    LRPipe = LogisticRegressionModel(df_ML)
    LSVCPipe = LSVCModel(df_ML)

    aval_TFIDF = getTFIDF(df_aval)
    environ_TFIDF = getTFIDF(df_environ)
    quality_TFIDF = getTFIDF(df_quality)
    safety_TFIDF = getTFIDF(df_safety)
    nonrel_TFIDF = getTFIDF(df_nonrel)
    aliasToLoc = getLatLngFarmersMarket(FM_INFO_DIR)

    res = {"Farmers_Market":[]}

    frames = parse_excel_by_class("/Users/jaewooklee/farmers_market/yelp/data/test_score_rev.xlsx")
    for x in frames:
        x.Label.replace(0, 1, inplace=True)

    df = pd.concat(frames)

    df_pos = df[df.Label == 1]
    df_neg = df[df.Label == -1]

    df_pos_avail = df_pos[df_pos.Category == AVAIL]
    df_neg_avail = df_neg[df_neg.Category == AVAIL]

    df_pos_en = df_pos[df_pos.Category == ENVIRON]
    df_neg_en = df_neg[df_neg.Category == ENVIRON]

    df_pos_qual = df_pos[df_pos.Category == QUAL]
    df_neg_qual = df_neg[df_neg.Category == QUAL]

    df_pos_safety = df_pos[df_pos.Category == SAFETY]
    df_neg_safety = df_neg[df_neg.Category == SAFETY]

    avail_pos_TFIDF = getTFIDF(df_pos_avail)
    avail_neg_TFIDF = getTFIDF(df_neg_avail)

    en_pos_TFIDF = getTFIDF(df_pos_en)
    en_neg_TFIDF = getTFIDF(df_neg_en)

    qual_pos_TFIDF = getTFIDF(df_pos_qual)
    qual_neg_TFIDF = getTFIDF(df_neg_qual)

    safety_pos_TFIDF = getTFIDF(df_pos_safety)
    safety_neg_TFIDF = getTFIDF(df_neg_safety)

    for file in os.listdir(SEPARATE_SENTENCE_DIR):
        sentences = []
        with open(SEPARATE_SENTENCE_DIR + file) as f:
            for i, line in enumerate(f):
                line = line.rstrip().lstrip()
                if i == 0 and line == "None":
                    break
                if line != "":
                    sentences.append(line)

        alias = file.split(".txt")[0]
        if len(sentences) > 0:
            fm = aliasToLoc[alias]
            count, overall = fm.num_reviews, fm.overall
            avail = []
            environ = []
            qual = []
            safety = []

            SGD_Pred = SGDPipe.predict(sentences)
            LR_Pred = LRPipe.predict(sentences)
            LSVC_Pred = LRPipe.predict(sentences)
            TFIDF_Pred = TFIDF_predict(sentences, aval_TFIDF, environ_TFIDF,
            quality_TFIDF, safety_TFIDF, nonrel_TFIDF)

            vote_pred = get_vote_pred(SGD_Pred, LSVC_Pred, LR_Pred, TFIDF_Pred)
            for sen, label in zip(sentences, vote_pred):
                if label == AVAIL:
                    avail.append(sen)
                elif label == ENVIRON:
                    environ.append(sen)
                elif label == QUAL:
                    qual.append(sen)
                elif label == SAFETY:
                    safety.append(sen)

            write_rev_per_asp(alias, 'avail', avail)
            write_rev_per_asp(alias, 'environ', environ)
            write_rev_per_asp(alias, 'qual', qual)
            write_rev_per_asp(alias, 'safety', safety)

            avail_sc = count_possitive_reviews(avail, AVAIL, avail_pos_TFIDF, avail_neg_TFIDF)
            environ_sc = count_possitive_reviews(environ, ENVIRON, en_pos_TFIDF, en_neg_TFIDF)
            qual_sc = count_possitive_reviews(qual, QUAL, qual_pos_TFIDF, qual_neg_TFIDF)
            safety_sc = count_possitive_reviews(safety, SAFETY, safety_pos_TFIDF, safety_neg_TFIDF)
            all_reviews = [avail_sc, environ_sc, qual_sc, safety_sc]
            review_scores = scoreing_algorithm(overall, all_reviews)

            round_scores = []
            for x in review_scores:
                if x is not None:
                    round_scores.append(round(x, 1))
                else:
                    round_scores.append(None)

            info = convert_info_to_dict(aliasToLoc[alias], round_scores)
            res["Farmers_Market"].append(info)

        else:
            scores = [None] * 4
            info = convert_info_to_dict(aliasToLoc[alias], scores)
            res["Farmers_Market"].append(info)

    with open("all_result.json", 'w') as f:
        f.write(json.dumps(res, indent=4))

get_all_score()

# overall = 4.5
# avail = (0, 0)
# environ = (0, 1)
# qual = (2, 0)
# safety = (0, 0)
# all_reviews = [avail, environ, qual, safety]
# print(scoreing_algorithm(overall, all_reviews))
