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

def main(n_iter=50):
    num_iter = n_iter
    sgd = np.zeros(4)
    svc = np.zeros(4)
    lsvc = np.zeros(4)
    mnb = np.zeros(4)
    cnb = np.zeros(4)
    bnb = np.zeros(4)
    lr = np.zeros(4)

    # Cross validation
    for i in range(0, num_iter):
        print(i)
        train, test = parse_csv(PATH)
        sgd += runSGD(train, test)
        svc += runSVC(train, test)
        lsvc += runLSVC(train, test)
        mnb += runMNB(train, test)
        cnb += runCNB(train, test)
        bnb += runBNB(train, test)
        lr += runLR(train, test)

    sgd /= num_iter
    svc /= num_iter
    lsvc /= num_iter
    mnb /= num_iter
    cnb /= num_iter
    bnb /= num_iter
    lr /= num_iter

    print(f"{sgd}\n{svc}\n{lsvc}\n{mnb}\n{cnb}\n{bnb}")
    analyze(sgd, svc, lsvc, lr, mnb, cnb, bnb)
