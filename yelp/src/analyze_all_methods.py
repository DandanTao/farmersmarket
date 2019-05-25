import RB_classification
import ML_classification
from csv_parser import parse_csv_by_class_v1
from test_all import get_vote_pred

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

PATH1="../data/yelp_labelling_1000.csv"
PATH2="../data/1000_more_yelp.csv"
PATH3="../data/2000_yelp_labeled.csv"

def draw_table(r1, r2, r3, r4, r5):
    import matplotlib.pyplot as plt
    methods = np.array(["SGD(ML)", "LSVC(ML)", "LR(ML)", "TFIDF(Staticical)", "VOTE"])
    measures = np.around(np.array([r1*100, r2*100, r3*100, r4*100, r5*100]), decimals=3)

    fig, ax = plt.subplots()
    ax.axis('off')

    cols = np.array(["Accuracy", "F1", "Recall", "Precision"])
    rows = methods

    table = plt.table(cellText=measures,
                    rowLabels=rows, #rowColours=['y' for x in methods],
                    colLabels=cols, colWidths=[0.15 for x in cols], #colColours=['c' for x in cols],
                    loc='center')

    plt.title("All Results")
    plt.subplots_adjust(left=0.2, top=0.8)
    plt.tight_layout()
    plt.show()

def analyze_all_classifier(num_iter, file_path):
    vote_a = np.zeros(4)
    sgd_a= np.zeros(4)
    lsvc_a = np.zeros(4)
    lr_a = np.zeros(4)
    tfidf_a = np.zeros(4)

    for _ in range(num_iter):
        # run_all(iter, PATH3, test, confusion_matrix=True, bar_graph=True)
        df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(PATH3)
        train1, test1 = train_test_split(df_aval, test_size=0.2)
        train2, test2 = train_test_split(df_environ, test_size=0.2)
        train3, test3 = train_test_split(df_quality, test_size=0.2)
        train4, test4 = train_test_split(df_safety, test_size=0.2)
        train5, test5 = train_test_split(df_nonrel, test_size=0.2)

        train=pd.concat([train1, train2, train3, train4, train5])
        test=pd.concat([test1, test2, test3, test4, test5])

        bow, tfidf, bow_pred, tfidf_pred = RB_classification.run_all(num_iter, file_path, test,
                                                                        confusion_matrix=False,
                                                                        bar_graph=False)


        sgd, lsvc, lr, cnb, sgd_pred, lsvc_pred, cnb_pred, lr_pred = ML_classification.run_all(cross_val=num_iter,
                                            train=train,
                                            test=test,
                                            analyze_metrics=False,
                                            confusion_matrix=False,
                                            file_path=file_path)


        true_label = test.Label

        vote_pred = get_vote_pred(sgd_pred, lsvc_pred, lr_pred, tfidf_pred)

        accuracy = accuracy_score(true_label, vote_pred)
        f1 = f1_score(true_label, vote_pred, average='weighted', labels=np.unique(vote_pred))
        recall = recall_score(true_label, vote_pred, average='weighted')
        precision = precision_score(true_label, vote_pred, average='weighted', labels=np.unique(vote_pred))

        tmp = np.array([accuracy, f1, recall, precision])
        vote_a += tmp
        sgd_a += sgd
        lsvc_a += lsvc
        lr_a += lr
        tfidf_a += tfidf

    vote_a /= num_iter
    sgd_a /= num_iter
    lsvc_a /= num_iter
    lr_a /= num_iter
    tfidf_a /= num_iter
    draw_table(sgd_a, lsvc_a, lr_a, tfidf_a, vote_a)


import sys
num_iter=1 if len(sys.argv) == 1 else int(sys.argv[1])

analyze_all_classifier(num_iter, PATH3)
