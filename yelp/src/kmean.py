#!/usr/bin/python3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from preprocess import *

PATH = "../data/separate_sentence.txt"
def kmean(file, true_k=5, max_iteration=100, init=1):
    """
    Reads file line by line and perform text clutering
    """
    vectorizer = TfidfVectorizer(preprocessor=preprocess)
    documents = []
    with open(file, "r") as f:
        for line in f:
            documents.append(line.rstrip())

    X = vectorizer.fit_transform(documents)
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=max_iteration, n_init=init)
    model.fit(X)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :10]:
            print("{} ".format(terms[ind]))
        print()

    return model, vectorizer

def test(model, vectorizer):
    ## ADD TESTS HERE
    return

def main():
    model, vectorizer = kmean(PATH)
    test(model, vectorizer)

if __name__ == '__main__':
    main()
