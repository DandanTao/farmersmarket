#!/usr/bin/python3

from models import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from ML_analyze import *
import warnings
import numpy as np
from RB_classification import *
from preprocess import preprocess, preprocessTFIDF
import matplotlib.pyplot as plt

PATH = "../data/test_score_rev.xlsx"

def parse_excel_by_class(file_name):
    df = pd.read_excel(file_name)
    df_aval = df[df.Category == 'availability']
    df_environ = df[df.Category =='environment']
    df_quality = df[df.Category =='quality']
    df_safety = df[df.Category =='safety']

    return (df_aval, df_environ, df_quality, df_safety)

def run_ML(model, train, test):
    pipe = model(train)
    pred = pipe.predict(test.Sentences)

    real_label = test.Label
    acc = accuracy_score(real_label, pred)
    f1 = f1_score(real_label, pred, average='weighted', labels=np.unique(pred))
    recall = recall_score(real_label, pred, average='weighted')
    precision = precision_score(real_label, pred, average='weighted', labels=np.unique(pred))
    metrics = [acc, f1, recall, precision]

    return (metrics, pred, real_label)

def runTFIDFRuleBase(file_path, train, test):
    """
    Deterministic rule based classification using TFIDF
    """
    df_neu = train[train.Label == 0]
    df_pos = train[train.Label == 1]
    df_neg = train[train.Label == -1]

    neu_TFIDF = getTFIDF(df_neu)
    pos_TFIDF = getTFIDF(df_pos)
    neg_TFIDF = getTFIDF(df_neg)

    count = 0
    correct = 0
    true_label = []
    pred_label = []
    for idx, row in test.iterrows():
        neu_sc = getCWScore(neu_TFIDF, row['Sentences'], preprocess)
        pos_sc = getCWScore(pos_TFIDF, row['Sentences'], preprocess)
        neg_sc = getCWScore(neg_TFIDF, row['Sentences'], preprocess)

        all_sc = [neg_sc, neu_sc, pos_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            predLabel = 0
        else:
            predLabel = all_sc.index(max_sc) - 1
        trueLabel = row["Label"]

        pred_label.append(predLabel)
        true_label.append(trueLabel)

    accuracy = accuracy_score(true_label, pred_label)
    f1 = f1_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))
    recall = recall_score(true_label, pred_label, average='weighted')
    precision = precision_score(true_label, pred_label, average='weighted', labels=np.unique(pred_label))

    metrics = [accuracy, f1, recall, precision]
    return metrics, pred_label, true_label

def merge_list(a, b):
    for x in b:
        a.append(x)
    return a

def main():
    frames = parse_excel_by_class(PATH)

    lsvc = np.zeros(4)
    lr = np.zeros(4)
    sgd = np.zeros(4)
    tfidf = np.zeros(4)

    lsvc_list = []
    lr_list = []
    sgd_list = []
    tfidf_list = []
    real_label = []

    import sys
    num_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(num_iter):
        print(i)
        for df in frames:
            train, test = train_test_split(df, test_size=0.2)

            lsvc_metrics, lsvc_pred, label = run_ML(LSVCModel, train, test)
            lr_metrics, lr_pred, _ = run_ML(LogisticRegressionModel, train, test)
            sgd_metrics, sgd_pred, _ = run_ML(SGDModel, train, test)
            tfidf_metrics, tfidf_pred, _ = runTFIDFRuleBase(PATH, train, test)

            lsvc += np.array(lsvc_metrics)
            lr += np.array(lr_metrics)
            sgd += np.array(sgd_metrics)
            tfidf += np.array(tfidf_metrics)

            lsvc_list = merge_list(lsvc_list, lsvc_pred)
            tfidf_list = merge_list(tfidf_list, tfidf_pred)
            lr_list = merge_list(lr_list, lr_pred)
            sgd_list = merge_list(sgd_list, sgd_pred)
            real_label = merge_list(real_label, label)

    lsvc /= (4 * num_iter)
    lr /= (4 * num_iter)
    sgd /= (4 * num_iter)
    tfidf /= (4 * num_iter)
    print(f"LSVC {lsvc}\nLR {lr}\nSGD {sgd}\nTFIDF {tfidf}")

    plot_confusion_matrix(real_label, lsvc_list, normalize=True, title="LSVC", cmap=plt.cm.Blues)
    plt.show()

    plot_confusion_matrix(real_label, lr_list, normalize=True, title="LR", cmap=plt.cm.Blues)
    plt.show()

    plot_confusion_matrix(real_label, sgd_list, normalize=True, title="SGD", cmap=plt.cm.Blues)
    plt.show()

    plot_confusion_matrix(real_label, tfidf_list, normalize=True, title="TFIDF", cmap=plt.cm.Blues)
    plt.show()

if __name__ == '__main__':
    main()
