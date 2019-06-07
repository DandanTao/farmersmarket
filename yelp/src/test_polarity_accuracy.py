#!/usr/bin/python3

import pandas as pd
import numpy as np
import os
from textblob import TextBlob
from preprocess import clean_text

REAL_LABEL_PATH = "../results/test_score.xlsx"

def get_accuracy(df):
    wrong = []
    correct = 0
    for i, row in df.iterrows():
        sen = row["Sentences"]
        sc = row["Score"]
        sentiment = TextBlob(clean_text(sen)).sentiment
        pol = 0
        if sentiment.polarity < 0:
            pol = -1
        elif sentiment.polarity > 0:
            pol = 1
        if int(sc) == int(pol):
            correct += 1
        else:
            wrong.append((sen, sc, pol))

    return (correct, len(df), wrong)


df = pd.read_excel(REAL_LABEL_PATH)

df_aval = df[df.Label == 'availability']
df_environ = df[df.Label =='environment']
df_quality = df[df.Label =='quality']
df_safety = df[df.Label =='safety']

c1, l1, w1 = get_accuracy(df_aval)
c2, l2, w2 = get_accuracy(df_environ)
c3, l3, w3 = get_accuracy(df_quality)
c4, l4, w4 = get_accuracy(df_safety)

w5 = w1+w2+w3+w4
overall = (c1+c2+c3+c4) / (l1+l2+l3+l4)
a1=c1/l1
a2=c2/l2
a3=c3/l3
a4=c4/l4
print(f"Availability: {a1}\tEnvironment: {a2}\tQuality: {a3}\tSafety: {a4}\
\nOverall: {overall}")
res={"Sentences":[x[0] for x in w5], "Labeled_score":[x[1] for x in w5], "Pred_score":[x[2] for x in w5]}
df_res = pd.DataFrame(data=res)
df_res.to_excel("../results/wrongly_classified.xlsx", index=False)
