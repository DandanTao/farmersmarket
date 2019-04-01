from train import *
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from analyze import *
import warnings
warnings.filterwarnings('ignore')
PATH="../data/yelp_labelling_1000.csv"
def runSVC(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = SVCModel(train)
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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

def runLR(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = LogisticRegressionModel(train)
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0].rstrip() for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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

def run_all(cross_val=10, plot=False, confusion_matrix=False):
    num_iter = cross_val
    sgd_a = np.zeros(4)
    svc_a = np.zeros(4)
    lsvc_a = np.zeros(4)
    mnb_a = np.zeros(4)
    cnb_a = np.zeros(4)
    bnb_a = np.zeros(4)
    lr_a = np.zeros(4)

    sgd_pred = np.array([])
    svc_pred = np.array([])
    lsvc_pred = np.array([])
    mnb_pred = np.array([])
    cnb_pred = np.array([])
    bnb_pred = np.array([])
    lr_pred = np.array([])

    sgd_real = np.array([])
    svc_real = np.array([])
    lsvc_real = np.array([])
    mnb_real = np.array([])
    cnb_real = np.array([])
    bnb_real = np.array([])
    lr_real = np.array([])

    # Cross validation
    for i in range(0, num_iter):
        print(i)
        train, test = parse_csv(PATH)
        sgd, sgd_p, sgd_r = runSGD(train, test)
        svc, svc_p, svc_r = runSVC(train, test)
        lsvc, lsvc_p, lsvc_r = runLSVC(train, test)
        mnb, mnb_p, mnb_r = runMNB(train, test)
        cnb, cnb_p, cnb_r = runCNB(train, test)
        bnb, bnb_p, bnb_r = runBNB(train, test)
        lr, lr_p, lr_r = runLR(train, test)

        sgd_a += sgd;
        svc_a += svc;
        lsvc_a += lsvc
        mnb_a += mnb
        cnb_a += cnb
        bnb_a += bnb
        lr_a += lr

        sgd_pred = np.append(sgd_pred, sgd_p)
        svc_pred = np.append(sgd_pred, svc_p)
        lsvc_pred = np.append(sgd_pred, lsvc_p)
        mnb_pred = np.append(sgd_pred, mnb_p)
        cnb_pred = np.append(sgd_pred, cnb_p)
        bnb_pred = np.append(sgd_pred, bnb_p)
        lr_pred = np.append(sgd_pred, lr_p)

        sgd_real = np.append(sgd_real, sgd_r)
        svc_real = np.append(sgd_real, svc_r)
        lsvc_real = np.append(sgd_real, lsvc_r)
        mnb_real = np.append(sgd_real, mnb_r)
        cnb_real = np.append(sgd_real, cnb_r)
        bnb_real = np.append(sgd_real, bnb_r)
        lr_real = np.append(sgd_real, lr_r)

    sgd_a /= num_iter
    svc_a /= num_iter
    lsvc_a /= num_iter
    mnb_a /= num_iter
    cnb_a /= num_iter
    bnb_a /= num_iter
    lr_a /= num_iter

    if plot:
        analyze(sgd_a, svc_a, lsvc_a, lr_a, mnb_a, cnb_a, bnb_a)
    if confusion_matrix:
        pred = [sgd_pred, svc_pred, lsvc_pred, mnb_pred, cnb_pred, bnb_pred, lr_pred]
        real = [sgd_real, svc_real, lsvc_real, mnb_real, cnb_real, bnb_real, lr_real]
        titles = ["SGD Normalized", "SVC Normalized", "LSVC Normalized", "MNB Normalized", "CNB Normalized", "BNB Normalized", "LR Normalized"]
        for p, r, t in zip(pred, real, titles):
            plot_confusion_matrix(p, r, normalize=True, title=t+" Confusion Matrix", cmap=plt.cm.Reds)
            plt.show()
