#!/usr/bin/python3
from models import *
from csv_parser import *
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from ML_analyze import *
import warnings
import numpy as np

warnings.filterwarnings('ignore')

PATH1="../data/yelp_labelling_1000.csv"
PATH2="../data/1000_more_yelp.csv"
PATH3="../data/2000_yelp_labeled.csv"
def runSVC(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = SVCModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runLSVC(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LSVCModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runLR(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LogisticRegressionModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runMNB(train, tests):
    """
    Wrapper function that uses training Multinomial Naive Bayes model to classify tests data
    """
    pipe = MultinomialNBModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runSGD(train, tests):
    """
    Wrapper function that uses training Stochastic Gradient Descent model to classify tests data
    """
    pipe = SGDModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runCNB(train, tests):
    """
    Wrapper function that uses training Complement Naive Bayes model to classify tests data
    """
    pipe = ComplementNBModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def runBNB(train, tests):
    """
    Wrapper function that uses training Bernoulli Naive Bayes model to classify tests data
    """
    pipe = BernoulliNBModel(train)

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    return metrics, pred_data, test_data

def consensus(lists_of_predict):
    c = []
    for i in range(0, np.shape(lists_of_predict[0])[0]):
        d=[]
        for p in lists_of_predict:
            d.append(p[i])
        c.append(max(d,key=d.count))
    return np.array(c)

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

def run_all(train=None, test=None):
    sgd, sgd_p, sgd_r = runSGD(train, test)
    lsvc, lsvc_p, lsvc_r = runLSVC(train, test)
    cnb, cnb_p, cnb_r = runCNB(train, test)
    lr, lr_p, lr_r = runLR(train, test)

    print(f"SGD {sgd}\nLSVC {lsvc}\nCNB {cnb}\nLR {lr}")

    return (sgd, lsvc, lr, cnb, sgd_p, lsvc_p, cnb_p, lr_p, sgd_r, lsvc_r, cnb_r, lr_r)

if __name__ == '__main__':
    import sys
    iter = 1 if len(sys.argv) == 1 else int(sys.argv[1])
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(PATH3)

    sgd_a = np.zeros(4)
    lsvc_a = np.zeros(4)
    cnb_a = np.zeros(4)
    lr_a = np.zeros(4)

    sgd_pred = np.array([])
    lsvc_pred = np.array([])
    cnb_pred = np.array([])
    lr_pred = np.array([])
    con_pred = np.array([])

    sgd_real = np.array([])
    lsvc_real = np.array([])
    cnb_real = np.array([])
    lr_real = np.array([])

    for _ in range(iter):
        train1, test1 = train_test_split(df_aval, test_size=0.2)
        train2, test2 = train_test_split(df_environ, test_size=0.2)
        train3, test3 = train_test_split(df_quality, test_size=0.2)
        train4, test4 = train_test_split(df_safety, test_size=0.2)
        train5, test5 = train_test_split(df_nonrel, test_size=0.2)

        train=pd.concat([train1, train2, train3, train4, train5])
        test=pd.concat([test1, test2, test3, test4, test5])
        sgd, lsvc, lr, cnb, sgd_p, lsvc_p, cnb_p, lr_p, sgd_r, lsvc_r, cnb_r, lr_r = run_all(train=train, test=test)

        sgd_a += sgd
        lsvc_a += lsvc
        cnb_a += cnb
        lr_a += lr

        sgd_pred = np.append(sgd_pred, sgd_p)
        lsvc_pred = np.append(lsvc_pred, lsvc_p)
        cnb_pred = np.append(cnb_pred, cnb_p)
        lr_pred = np.append(lr_pred, lr_p)

        sgd_real = np.append(sgd_real, sgd_r)
        lsvc_real = np.append(lsvc_real, lsvc_r)
        cnb_real = np.append(cnb_real, cnb_r)
        lr_real = np.append(lr_real, lr_r)

    sgd_a /= iter
    lsvc_a /= iter
    cnb_a /= iter
    lr_a /= iter

    analyze(sgd_a, lsvc_a, lr_a, cnb_a)
    pred = [sgd_pred, lsvc_pred, cnb_pred, lr_pred]
    real = [sgd_real, lsvc_real, cnb_real, lr_real]
    titles = ["SGD Normalized", "LSVC Normalized", "CNB Normalized", "LR Normalized"]
    for p, r, t in zip(pred, real, titles):
        plot_confusion_matrix(p, r, normalize=True, title=t+" Confusion Matrix", cmap=plt.cm.Reds)
        plt.show()

    print_stat_to_file([sgd_a, lsvc_a, cnb_a, lr_a], ["SGD", "LSVC", "CNB", "LR"])
