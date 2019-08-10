#!/usr/bin/python3
import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from ML_analyze import *
import warnings
import excel_parser
import sys
from sklearn.model_selection import train_test_split
import preprocessor as p
import preprocessor_v0
from cleanup_tweet import *

warnings.filterwarnings('ignore')

BUSINESS_TRAIN = '/Users/jaewooklee/farmers_market/twitter/data/bus_nobus_cleaned.xlsx'
LABELING_TRAIN = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned.xlsx'
LABEL_NEW = '/Users/jaewooklee/farmers_market/twitter/data/twitter_1000_labelling_cleaned_new.xlsx'
def runSVC(train, tests, ngram):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = SVCModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runLSVC(train, tests, ngram):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LSVCModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runLR(train, tests, ngram):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LogisticRegressionModel(train, ngram)
    # pred_data = pipe.predict([x[0] for x in tests])
    # test_data = np.array([x[1].rstrip() for x in tests])

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runMNB(train, tests, ngram):
    """
    Wrapper function that uses training Multinomial Naive Bayes model to classify tests data
    """
    pipe = MultinomialNBModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runSGD(train, tests, ngram):
    """
    Wrapper function that uses training Stochastic Gradient Descent model to classify tests data
    """
    pipe = SGDModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runCNB(train, tests, ngram):
    """
    Wrapper function that uses training Complement Naive Bayes model to classify tests data
    """
    pipe = ComplementNBModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runBNB(train, tests, ngram):
    """
    Wrapper function that uses training Bernoulli Naive Bayes model to classify tests data
    """
    pipe = BernoulliNBModel(train, ngram)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]


    return metrics, pred_data, test_data

def run_all(train, test, ngram):
    lsvc, lsvc_p, lsvc_r = runLSVC(train, test, ngram)
    sgd, sgd_p, sgd_r = runSGD(train, test, ngram)
    lr, lr_p, lr_r = runLR(train, test, ngram)
    cnb, cnb_p, cnb_r = runCNB(train, test, ngram)


    print(f"SGD {sgd}\nLSVC {lsvc}\nCNB {cnb}\nLR {lr}")

    return (sgd, lsvc, lr, cnb, sgd_p, lsvc_p, lr_p, cnb_p, sgd_r, lsvc_r, lr_r, cnb_r) # prediction data for vote

def split_test_bus_vs_nobus(df_bus, df_nobus):
    train1, test1 = train_test_split(df_bus, test_size=0.2)
    train2, test2 = train_test_split(df_nobus, test_size=0.2)

    train=pd.concat([train1, train2])
    test=pd.concat([test1, test2])

    return (train, test)

def split_test_labels(df_avail, df_environ, df_qual, df_safety, df_NR):
    train1, test1 = train_test_split(df_avail, test_size=0.2)
    train2, test2 = train_test_split(df_environ, test_size=0.2)
    train3, test3 = train_test_split(df_qual, test_size=0.2)
    train4, test4 = train_test_split(df_safety, test_size=0.2)
    train5, test5 = train_test_split(df_NR, test_size=0.2)

    train=pd.concat([train1, train2, train3, train4, train5])
    test=pd.concat([test1, test2, test3, test4, test5])

    # train=pd.concat([train1, train2, train3, train4])
    # test=pd.concat([test1, test2, test3, test4])

    return (train, test)

def preprocess_df_yelp(df):
    for i, content in enumerate(df.Sentences):
        cleaned = clean_tweets(p.clean(content).lower())
        df.at[i, 'Sentences'] = preprocessor_v0.twitter_preprocess(cleaned)

    return df

def print_stat_to_file(stats, names):
    res_json = {}
    for stat, name in zip(stats, names):
        res_json[name] = {}
        res_json[name]['accuracy'] = round(stat[0], 4)
        res_json[name]['f1'] = round(stat[1], 4)
        res_json[name]['recall'] = round(stat[2], 4)
        res_json[name]['precision'] = round(stat[3], 4)
    import json
    with open("metrics.json", 'w') as f:
        f.write(json.dumps(res_json, indent=4))

def main():
    num_iter = 10

    sgd_a = np.zeros(4)
    lsvc_a = np.zeros(4)
    cnb_a = np.zeros(4)
    lr_a = np.zeros(4)

    sgd_pred = np.array([])
    lsvc_pred = np.array([])
    cnb_pred = np.array([])
    lr_pred = np.array([])

    sgd_real = np.array([])
    lsvc_real = np.array([])
    cnb_real = np.array([])
    lr_real = np.array([])
    if 'NR' in sys.argv:
        df_NR, df_R = excel_parser.parse_excel_NR()

        for epoch in range(num_iter):
            train, test = split_test_bus_vs_nobus(df_NR, df_R)
            print(f"Epoch {epoch + 1}")
            ngram = (1, 1)
            sgd, lsvc, lr, cnb, sgd_p, lsvc_p, lr_p, cnb_p, sgd_r, lsvc_r, lr_r, cnb_r = run_all(train, test, ngram)

            sgd_a += sgd;
            lsvc_a += lsvc
            cnb_a += cnb
            lr_a += lr

            sgd_pred = np.append(sgd_pred, sgd_p)
            lsvc_pred = np.append(lsvc_pred, lsvc_p)
            lr_pred = np.append(lr_pred, lr_p)
            cnb_pred = np.append(cnb_pred, cnb_p)

            sgd_real = np.append(sgd_real, sgd_r)
            lsvc_real = np.append(lsvc_real, lsvc_r)
            lr_real = np.append(lr_real, lr_r)
            cnb_real = np.append(cnb_real, cnb_r)

        sgd_a /= num_iter
        lsvc_a /= num_iter
        lr_a /= num_iter
        cnb_a /= num_iter

        print(f"FINAL:\nSGD {sgd_a}\nLSVC {lsvc_a}\nCNB {cnb_a}\nLR {lr_a}")

    elif 'bus' in sys.argv:
        df_bus, df_nobus = excel_parser.parse_excel_bus_nobus()

        for epoch in range(num_iter):
            train, test = split_test_bus_vs_nobus(df_bus, df_nobus)
            print(f"Epoch {epoch + 1}")
            ngram = (1, 3)
            sgd, lsvc, lr, cnb, sgd_p, lsvc_p, lr_p, cnb_p, sgd_r, lsvc_r, lr_r, cnb_r = run_all(train, test, ngram)

            sgd_a += sgd;
            lsvc_a += lsvc
            cnb_a += cnb
            lr_a += lr

            sgd_pred = np.append(sgd_pred, sgd_p)
            lsvc_pred = np.append(lsvc_pred, lsvc_p)
            lr_pred = np.append(lr_pred, lr_p)
            cnb_pred = np.append(cnb_pred, cnb_p)

            sgd_real = np.append(sgd_real, sgd_r)
            lsvc_real = np.append(lsvc_real, lsvc_r)
            lr_real = np.append(lr_real, lr_r)
            cnb_real = np.append(cnb_real, cnb_r)

        sgd_a /= num_iter
        lsvc_a /= num_iter
        lr_a /= num_iter
        cnb_a /= num_iter

        print(f"FINAL:\nSGD {sgd_a}\nLSVC {lsvc_a}\nCNB {cnb_a}\nLR {lr_a}")

    else:
        # df_aval, df_environ, df_quality, df_safety, df_nonrel \
        #                 = excel_parser.parse_yelp()
        # frames = [df_aval, df_environ, df_quality, df_safety, df_nonrel]
        # train1 = pd.concat(frames)

        # train1 = preprocess_df_yelp(train1)

        df_NR, df_avail, df_environ, df_qual, df_safety = excel_parser.parse_excel_labels()

        for epoch in range(num_iter):
            train, test = split_test_labels(df_avail, df_environ, df_qual, df_safety, df_NR)
            # train = pd.concat([train1, train])
            # test = pd.concat([df_NR, df_avail, df_environ, df_qual, df_safety])
            # train = train1
            print(f"Epoch {epoch + 1}")
            ngram = (1, 1)
            sgd, lsvc, lr, cnb, sgd_p, lsvc_p, lr_p, cnb_p, sgd_r, lsvc_r, lr_r, cnb_r = run_all(train, test, ngram)

            sgd_a += sgd;
            lsvc_a += lsvc
            lr_a += lr
            cnb_a += cnb

            sgd_pred = np.append(sgd_pred, sgd_p)
            lsvc_pred = np.append(lsvc_pred, lsvc_p)
            lr_pred = np.append(lr_pred, lr_p)
            cnb_pred = np.append(cnb_pred, cnb_p)

            sgd_real = np.append(sgd_real, sgd_r)
            lsvc_real = np.append(lsvc_real, lsvc_r)
            lr_real = np.append(lr_real, lr_r)
            cnb_real = np.append(cnb_real, cnb_r)

        sgd_a /= num_iter
        lsvc_a /= num_iter
        lr_a /= num_iter
        cnb_a /= num_iter

        print(f"FINAL:\nSGD {sgd_a}\nLSVC {lsvc_a}\nCNB {cnb_a}\nLR {lr_a}")

    ## METRICS
    analyze(sgd_a, lsvc_a, lr_a, cnb_a)

    ## CONFUSION MATRIX
    pred = [sgd_pred, lsvc_pred, cnb_pred, lr_pred]
    real = [sgd_real, lsvc_real, cnb_real, lr_real]
    titles = ["SGD Normalized", "LSVC Normalized", "CNB Normalized", "LR Normalized"]
    for p, r, t in zip(pred, real, titles):
        plot_confusion_matrix(p, r, normalize=True, title=t+" Confusion Matrix", cmap=plt.cm.Reds)
        plt.show()

    print_stat_to_file([sgd_a, lsvc_a, cnb_a, lr_a], ["SGD", "LSVC", "CNB", "LR"])


if __name__ == '__main__':
    # Label 2009 to 2013
    # 20 top terms for each model and the weights and idf
    main()

# sklearn varianceThreshold for feature selection with threshhold of 1.5/N
