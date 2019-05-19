import matplotlib.pyplot as plt
import numpy as np


def RB_bar_graph(BOW_metrics, TFIDF_metrics):
    n_groups = 4
    metrics = ('Accuracy', 'F1', 'Recall', 'Precision')

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.3

    bow_bar = plt.bar(index, BOW_metrics, bar_width, alpha=0.8, color='m', label='BOW')
    tfidf_bar = plt.bar(index+bar_width, TFIDF_metrics, bar_width, alpha=0.8, color='c', label='TFIDF')

    plt.ylim(0, 1)
    plt.ylabel('Scores')
    plt.title('Metric Scores for RB_Classifications')
    plt.xticks(index + bar_width/2, metrics)
    plt.legend()

    plt.tight_layout()
    plt.show()
