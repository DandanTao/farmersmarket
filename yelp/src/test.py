from train import *
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from analyze import *
import warnings
from pprint import pprint
from preprocess import preprocess, preprocessTFIDF
import math
import numpy as np

import LDA

warnings.filterwarnings('ignore')
PATH="../data/yelp_labelling_1000.csv"
def runSVC(train, tests):
    """
    Wrapper function that uses training Support Vector Machine model to classify tests data
    """
    pipe = SVCModel(train)
    pred_data = pipe.predict([x[0] for x in tests])
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
    pred_data = pipe.predict([x[0] for x in tests])
    test_data = np.array([x[1].rstrip() for x in tests])

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
    pred_data = pipe.predict([x[0] for x in tests])
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
    pred_data = pipe.predict([x[0] for x in tests])
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
    pred_data = pipe.predict([x[0] for x in tests])
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
    pred_data = pipe.predict([x[0] for x in tests])
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
    pred_data = pipe.predict([x[0] for x in tests])
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

def wordCount(df, preprocessor):
    res = {}
    count = 0
    for row in df.iterrows():
        for word in preprocessor(row[1]["Sentences"]).split(" "):
            if word in res:
                res[word] += 1
            else:
                res[word] = 1
            count += 1
    return (res, count)

def wordCountRuleBase(file_path):
    """
    BOG model
    """
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class(file_path)

    aval = sorted(wordCount(df_aval, preprocess)[0].items(), key=lambda x: x[1], reverse=True)
    environ = sorted(wordCount(df_environ, preprocess)[0].items(), key=lambda x: x[1], reverse=True)
    qual = sorted(wordCount(df_quality, preprocess)[0].items(), key=lambda x: x[1], reverse=True)
    safety = sorted(wordCount(df_safety, preprocess)[0].items(), key=lambda x: x[1], reverse=True)
    nonrel = sorted(wordCount(df_nonrel, preprocess)[0].items(), key=lambda x: x[1], reverse=True)
    return (aval, environ, qual, safety, nonrel)

def TFNormalize(wordcount, num_terms):
    res = {}
    total = 0

    for i, (k, v) in enumerate(wordcount[:num_terms]):
        total += v

    for i in range(0, num_terms):
        res[wordcount[i][0]] = wordcount[i][1] / total
    return res

def getCWScore(TFDict, sen, preprocessor):
    sum = 0
    for word in preprocessor(sen).split(" "):
        if word in TFDict:
            sum += TFDict[word]
    return sum

def runWordCountRuleBase(file_path, num_terms=20):
    aval, environ, qual, safety, nonrel = wordCountRuleBase(file_path)
    aval_dict = TFNormalize(aval, num_terms)
    environ_dict = TFNormalize(environ, num_terms)
    qual_dict = TFNormalize(qual, num_terms)
    safety_dict = TFNormalize(safety, num_terms)
    nonrel_dict = TFNormalize(nonrel, num_terms)

    df = pd.read_csv(file_path)
    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']
    _, test = train_test_split(df, test_size=0.2)
    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'Non-relevant'}
    count = 0
    correct = 0
    for idx, row in test.iterrows():
        trueLabel = row["Label"]
        aval_sc = getCWScore(aval_dict, row['Sentences'], preprocess)
        environ_sc = getCWScore(environ_dict, row['Sentences'], preprocess)
        qual_sc = getCWScore(qual_dict, row['Sentences'], preprocess)
        safety_sc = getCWScore(safety_dict, row['Sentences'], preprocess)
        nonrel_sc = getCWScore(nonrel_dict, row['Sentences'], preprocess)
        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            continue
        count += 1
        if idx_to_Label[all_sc.index(max_sc)] == trueLabel:
            correct += 1
    print(f"BOW MODEL RULE_BASED CLASSIFICATION: {correct / count}")
    return correct / count

def computeTF(df):
    dict, count = wordCount(df, preprocessTFIDF)
    for k, v in dict.items():
        dict[k] /= count
    return dict


def computeIDF(df):
    N = len(df)
    words = set()
    all_sen = []
    for i, row in df.iterrows():
        sen = preprocessTFIDF(row["Sentences"])
        all_sen.append(sen)
        for word in sen.split(" "):
            words.add(word)

    res = {}
    for word in words:
        n = 1 #smoothing
        for sen in all_sen:
            if word in sen:
                n += 1
        res[word] = np.log(N / n)

    return res

def computeTFIDF(TF, IDF):
    res = {}
    for k, v in TF.items():
        res[k] = v * IDF[k]
    return res

def runTFIDFRuleBase(file_path):
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class(file_path)
    aval_TFDict = computeTF(df_aval)
    environ_TFDict = computeTF(df_environ)
    quality_TFDict = computeTF(df_quality)
    safety_TFDict = computeTF(df_safety)
    nonrel_TFDict = computeTF(df_nonrel)

    aval_IDFDict = computeIDF(df_aval)
    environ_IDFDict = computeIDF(df_environ)
    quality_IDFDict = computeIDF(df_quality)
    safety_IDFDict = computeIDF(df_safety)
    nonrel_IDFDict = computeIDF(df_nonrel)

    aval_TFIDF = computeTFIDF(aval_TFDict, aval_IDFDict)
    environ_TFIDF = computeTFIDF(environ_TFDict, environ_IDFDict)
    quality_TFIDF = computeTFIDF(quality_TFDict, quality_IDFDict)
    safety_TFIDF = computeTFIDF(safety_TFDict, safety_IDFDict)
    nonrel_TFIDF = computeTFIDF(nonrel_TFDict, nonrel_IDFDict)

    df = pd.read_csv(file_path)
    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']
    _, test = train_test_split(df, test_size=0.2)
    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'Non-relevant'}
    count = 0
    correct = 0
    for idx, row in test.iterrows():
        trueLabel = row["Label"]
        aval_sc = getCWScore(aval_TFIDF, row['Sentences'], preprocess)
        environ_sc = getCWScore(environ_TFIDF, row['Sentences'], preprocess)
        qual_sc = getCWScore(quality_TFIDF, row['Sentences'], preprocess)
        safety_sc = getCWScore(safety_TFIDF, row['Sentences'], preprocess)
        nonrel_sc = getCWScore(nonrel_TFIDF, row['Sentences'], preprocess)
        all_sc = [aval_sc, environ_sc, qual_sc, safety_sc, nonrel_sc]
        max_sc = max(all_sc)
        if max_sc == 0:
            continue
        count += 1
        if idx_to_Label[all_sc.index(max_sc)] == trueLabel:
            correct += 1
    print(f"TFIDF MODEL RULE_BASED CLASSIFICATION: {correct / count}")
    return correct / count


def runLDARuleBase(file_path):
    import gensim
    from gensim import corpora, models
    df_aval, df_environ, df_quality, df_safety, df_nonrel = parse_csv_by_class(file_path)

    aval_processed_docs = df_aval["Sentences"].map(LDA.preprocess)
    environ_processed_docs = df_environ["Sentences"].map(LDA.preprocess)
    quality_processed_docs = df_quality["Sentences"].map(LDA.preprocess)
    safety_processed_docs = df_safety["Sentences"].map(LDA.preprocess)
    nonrel_processed_docs = df_nonrel["Sentences"].map(LDA.preprocess)

    aval_dictionary = gensim.corpora.Dictionary(aval_processed_docs)
    environ_dictionary = gensim.corpora.Dictionary(environ_processed_docs)
    quality_dictionary = gensim.corpora.Dictionary(quality_processed_docs)
    safety_dictionary = gensim.corpora.Dictionary(safety_processed_docs)
    nonrel_dictionary = gensim.corpora.Dictionary(nonrel_processed_docs)

    aval_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    environ_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    quality_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    # NOTE: No filtering since there are no sentences related to safety in the first place
    # safety_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    nonrel_dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

    aval_bow_corpus = [aval_dictionary.doc2bow(doc) for doc in aval_processed_docs]
    environ_bow_corpus = [environ_dictionary.doc2bow(doc) for doc in environ_processed_docs]
    quality_bow_corpus = [quality_dictionary.doc2bow(doc) for doc in quality_processed_docs]
    safety_bow_corpus = [safety_dictionary.doc2bow(doc) for doc in safety_processed_docs]
    nonrel_bow_corpus = [nonrel_dictionary.doc2bow(doc) for doc in nonrel_processed_docs]

    aval_tfidf = models.TfidfModel(aval_bow_corpus)
    environ_tfidf = models.TfidfModel(environ_bow_corpus)
    quality_tfidf = models.TfidfModel(quality_bow_corpus)
    safety_tfidf = models.TfidfModel(safety_bow_corpus)
    nonrel_tfidf = models.TfidfModel(nonrel_bow_corpus)

    aval_corpus_tfidf = aval_tfidf[aval_bow_corpus]
    environ_corpus_tfidf = environ_tfidf[environ_bow_corpus]
    quality_corpus_tfidf = quality_tfidf[quality_bow_corpus]
    safety_corpus_tfidf = safety_tfidf[safety_bow_corpus]
    nonrel_corpus_tfidf = nonrel_tfidf[nonrel_bow_corpus]

    aval_lda_model = gensim.models.LdaMulticore(aval_bow_corpus, num_topics=10, id2word=aval_dictionary, passes=2, workers=2)
    environ_lda_model = gensim.models.LdaMulticore(environ_bow_corpus, num_topics=10, id2word=environ_dictionary, passes=2, workers=2)
    quality_lda_model = gensim.models.LdaMulticore(quality_bow_corpus, num_topics=10, id2word=quality_dictionary, passes=2, workers=2)
    safety_lda_model = gensim.models.LdaMulticore(safety_bow_corpus, num_topics=10, id2word=safety_dictionary, passes=2, workers=2)
    nonrel_lda_model = gensim.models.LdaMulticore(nonrel_bow_corpus, num_topics=10, id2word=nonrel_dictionary, passes=2, workers=2)

    aval_lda_model_tfidf = gensim.models.LdaMulticore(aval_corpus_tfidf, num_topics=10, id2word=aval_dictionary, passes=2, workers=4)
    environ_lda_model_tfidf = gensim.models.LdaMulticore(environ_corpus_tfidf, num_topics=10, id2word=environ_dictionary, passes=2, workers=4)
    quality_lda_model_tfidf = gensim.models.LdaMulticore(quality_corpus_tfidf, num_topics=10, id2word=quality_dictionary, passes=2, workers=4)
    safety_lda_model_tfidf = gensim.models.LdaMulticore(safety_corpus_tfidf, num_topics=10, id2word=safety_dictionary, passes=2, workers=4)
    nonrel_lda_model_tfidf = gensim.models.LdaMulticore(nonrel_corpus_tfidf, num_topics=10, id2word=nonrel_dictionary, passes=2, workers=4)

    df = pd.read_csv(file_path)
    df.drop(["Index", "Unnamed: 5", "Notes", "Words"], axis=1, inplace=True)
    df=df.fillna("Non-relevant")
    df = df[df.Label != 'covinience']
    df = df[df.Label != 'safety?']
    _, test = train_test_split(df, test_size=0.2)
    idx_to_Label = {0:'availability', 1:'environment', 2:'quality', 3:'safety', 4:'Non-relevant'}
    count = 0
    bow_correct = 0
    tfidf_correct = 0
    for idx, row in test.iterrows():
        trueLabel = row["Label"]
        text = row['Sentences']
        scores = []
        scores.append(LDA.score_doc(text, aval_dictionary, aval_lda_model, aval_lda_model_tfidf))
        scores.append(LDA.score_doc(text, environ_dictionary, environ_lda_model, environ_lda_model_tfidf))
        scores.append(LDA.score_doc(text, quality_dictionary, quality_lda_model, quality_lda_model_tfidf))
        scores.append(LDA.score_doc(text, safety_dictionary, safety_lda_model, safety_lda_model_tfidf))
        scores.append(LDA.score_doc(text, nonrel_dictionary, nonrel_lda_model, nonrel_lda_model_tfidf))

        bow_sc = [x[0] for x in scores]
        tfidf_sc = [x[1] for x in scores]

        count += 1
        if idx_to_Label[bow_sc.index(max(bow_sc))] == trueLabel:
            bow_correct += 1

        if idx_to_Label[tfidf_sc.index(max(tfidf_sc))] == trueLabel:
            tfidf_correct += 1

    print(f"LDA TOPIC MODELING WITH BOW: {bow_correct / count}")
    print(f"LDA TOPIC MODELING WITH TFIDF: {tfidf_correct / count}")

    return (bow_correct / count, tfidf_correct / count)

def consensus(lists_of_predict):
    c = []
    for i in range(0, np.shape(lists_of_predict[0])[0]):
        d=[]
        for p in lists_of_predict:
            d.append(p[i])
        c.append(max(d,key=d.count))
    return np.array(c)


def run_all(cross_val=10, analyze_metrics=False, confusion_matrix=False, parser=parse_csv_relevant_non_relevant):
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
        train, test = parser(PATH)
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

    print(accuracy_score(sgd_real, con_pred))
    if analyze_metrics:
        # analyze(sgd_a, lsvc_a, lr_a, mnb_a, cnb_a, bnb_a)
        print(f"SGD {sgd_a}\nLSVC {lsvc_a}\nCNB {cnb_a} LR {lr_a}")
        analyze(sgd_a, lsvc_a, lr_a, cnb_a)
    if confusion_matrix:
        pred = [sgd_pred, lsvc_pred, cnb_pred, lr_pred, con_pred]
        real = [sgd_real, lsvc_real, cnb_real, lr_real, sgd_real]
        titles = ["SGD Normalized", "LSVC Normalized", "CNB Normalized", "LR Normalized", "Consensus Normalized"]
        for p, r, t in zip(pred, real, titles):
            plot_confusion_matrix(p, r, normalize=True, title=t+" Confusion Matrix", cmap=plt.cm.Reds)
            plt.show()

import sys
# run_all(cross_val=int(sys.argv[1]),
#         analyze_metrics=True,
#         confusion_matrix=True,
#         parser=parse_csv_relevant_non_relevant)

bow = []
tfidf = []
bow_lda = []
tfidf_lda = []
for i in range(20):
    bow.append(runWordCountRuleBase(PATH, num_terms=30))
    tfidf.append(runTFIDFRuleBase(PATH))
    b, t = runLDARuleBase(PATH)
    bow_lda += [b]
    tfidf_lda += [t]
    print()

def stat(model, stats):
    stats.sort()
    print(f"{model}\tMAX {max(stats)}\tMEDIAN {stats[len(stats)//2]}\tMEAN {sum(stats)/len(stats)}")

print("FINAL STAT")
stat("BOW", bow)
stat("TFIDF", tfidf)
stat("LDA_BOW", bow_lda)
stat("LDA_TFIDF", tfidf_lda)
