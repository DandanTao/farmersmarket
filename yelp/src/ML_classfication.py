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

    # print ("SVC Accuracy:", accuracy)
    # print ("SVC F Score:", f1)
    # print ("SVC Recall:", recall)
    # print ("SVC Precision:", precision)

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

    # print ("LSVC Accuracy:", accuracy)
    # print ("LSVC F Score:", f1)
    # print ("LSVC Recall:", recall)
    # print ("LSVC Precision:", precision)

    return metrics, pred_data, test_data

def runLR(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LogisticRegressionModel(train)
    # pred_data = pipe.predict([x[0] for x in tests])
    # test_data = np.array([x[1].rstrip() for x in tests])

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    # print ("SVC Accuracy:", accuracy)
    # print ("SVC F Score:", f1)
    # print ("SVC Recall:", recall)
    # print ("SVC Precision:", precision)

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

    # print ("MNB Accuracy:", accuracy)
    # print ("MNB F Score:", f1)
    # print ("MNB Recall:", recall)
    # print ("MNB Precision:", precision)

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

    # print ("SGD Accuracy:", accuracy)
    # print ("SGD F Score:", f1)
    # print ("SGD Recall:", recall)
    # print ("SGD Precision:", precision)

    return metrics, pred_data, test_data

def runCNB(train, tests):
    """
    Wrapper function that uses training Complement Naive Bayes model to classify tests data
    """
    pipe = ComplementNBModel(train)
    # pred_data = pipe.predict([x[0] for x in tests])
    # test_data = np.array([x[1].rstrip() for x in tests])

    pred_data = pipe.predict(tests.Sentences)
    test_data = tests.Label

    accuracy = accuracy_score(test_data, pred_data)
    f1 = f1_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))
    recall = recall_score(test_data, pred_data, average='weighted')
    precision = precision_score(test_data, pred_data, average='weighted', labels=np.unique(pred_data))

    metrics = [accuracy, f1, recall, precision]

    # print ("CNB Accuracy:", accuracy)
    # print ("CNB F Score:", f1)
    # print ("CNB Recall:", recall)
    # print ("CNB Precision:", precision)

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

    # print ("BNB Accuracy:", accuracy)
    # print ("BNB F Score:", f1)
    # print ("BNB Recall:", recall)
    # print ("BNB Precision:", precision)

    return metrics, pred_data, test_data

def consensus(lists_of_predict):
    c = []
    for i in range(0, np.shape(lists_of_predict[0])[0]):
        d=[]
        for p in lists_of_predict:
            d.append(p[i])
        c.append(max(d,key=d.count))
    return np.array(c)

def run_all(cross_val=10, analyze_metrics=False, confusion_matrix=False, file_path=None):
    num_iter = cross_val
    sgd_a = np.zeros(4)
    lsvc_a = np.zeros(4)
    # mnb_a = np.zeros(4)
    cnb_a = np.zeros(4)
    # bnb_a = np.zeros(4)
    lr_a = np.zeros(4)

    sgd_pred = np.array([])
    lsvc_pred = np.array([])
    # mnb_pred = np.array([])
    cnb_pred = np.array([])
    # bnb_pred = np.array([])
    lr_pred = np.array([])
    con_pred = np.array([])

    sgd_real = np.array([])
    lsvc_real = np.array([])
    # mnb_real = np.array([])
    cnb_real = np.array([])
    # bnb_real = np.array([])
    lr_real = np.array([])

    con_acc=0
    # Cross validation
    for i in range(0, num_iter):
        # train, test = parser(PATH)
        df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class_v1(file_path)

        train1, test1 = train_test_split(df_aval, test_size=0.2)
        train2, test2 = train_test_split(df_environ, test_size=0.2)
        train3, test3 = train_test_split(df_quality, test_size=0.2)
        train4, test4 = train_test_split(df_safety, test_size=0.2)
        train5, test5 = train_test_split(df_nonrel, test_size=0.2)

        # df = pd.concat([df_aval, df_environ, df_quality, df_safety, df_nonrel])
        # train, test = train_test_split(df, test_size=0.2)
        # print(df.Label.unique())

        train=pd.concat([train1, train2, train3, train4, train5])
        test=pd.concat([test1, test2, test3, test4, test5])

        sgd, sgd_p, sgd_r = runSGD(train, test)
        lsvc, lsvc_p, lsvc_r = runLSVC(train, test)
        # mnb, mnb_p, mnb_r = runMNB(train, test)
        cnb, cnb_p, cnb_r = runCNB(train, test)
        # bnb, bnb_p, bnb_r = runBNB(train, test)
        lr, lr_p, lr_r = runLR(train, test)

        sgd_a += sgd;
        lsvc_a += lsvc
        # mnb_a += mnb
        cnb_a += cnb
        # bnb_a += bnb
        lr_a += lr

        sgd_pred = np.append(sgd_pred, sgd_p)
        lsvc_pred = np.append(lsvc_pred, lsvc_p)
        # mnb_pred = np.append(mnb_pred, mnb_p)
        cnb_pred = np.append(cnb_pred, cnb_p)
        # bnb_pred = np.append(bnb_pred, bnb_p)
        lr_pred = np.append(lr_pred, lr_p)
        con_pred = np.append(con_pred, consensus([lsvc_p, cnb_p, lr_p]))

        sgd_real = np.append(sgd_real, sgd_r)
        lsvc_real = np.append(lsvc_real, lsvc_r)
        # mnb_real = np.append(mnb_real, mnb_r)
        cnb_real = np.append(cnb_real, cnb_r)
        # bnb_real = np.append(bnb_real, bnb_r)
        lr_real = np.append(lr_real, lr_r)
    sgd_a /= num_iter
    lsvc_a /= num_iter
    # mnb_a /= num_iter
    cnb_a /= num_iter
    # bnb_a /= num_iter
    lr_a /= num_iter

    print(f"SGD {sgd_a}\nLSVC {lsvc_a}\nCNB {cnb_a} LR {lr_a}")

    if analyze_metrics:
        analyze(sgd_a, lsvc_a, lr_a, cnb_a)
    if confusion_matrix:
        pred = [sgd_pred, lsvc_pred, cnb_pred, lr_pred, con_pred]
        real = [sgd_real, lsvc_real, cnb_real, lr_real, sgd_real]
        titles = ["SGD Normalized", "LSVC Normalized", "CNB Normalized", "LR Normalized", "Consensus Normalized"]
        for p, r, t in zip(pred, real, titles):
            plot_confusion_matrix(p, r, normalize=True, title=t+" Confusion Matrix", cmap=plt.cm.Reds)
            plt.show()

def main():
    import sys
    iter = 1 if len(sys.argv) == 1 else int(sys.argv[1])

    run_all(cross_val=iter,
            analyze_metrics=True,
            confusion_matrix=True,
            file_path=PATH3)

if __name__ == '__main__':
    main()
