import matplotlib.pyplot as plt
import numpy as np
def analyze(SGD_metrics, SVC_metrics, LSVC_metrics, LR_metrics, MNB_metrics, CNB_metrics, BNB_metrics):
    n_groups = 4
    metrics = ('Accuracy', 'F1', 'Recall', 'Precision')

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.1
    opacity = 0.8

    sgd_bar = plt.bar(index, SGD_metrics, bar_width, alpha=opacity, color='b', label='SGD')

    svc_bar = plt.bar(index + bar_width, SVC_metrics, bar_width, alpha=opacity, color='g', label='SVC')

    lsvc_bar = plt.bar(index + bar_width * 2, LSVC_metrics, bar_width, alpha=opacity, color='k', label='LSVC')

    lr_bar = plt.bar(index + bar_width * 3, LR_metrics, bar_width, alpha=opacity, color='m', label='LR')

    mnb_bar = plt.bar(index + bar_width * 4, MNB_metrics, bar_width, alpha=opacity, color='r', label='MNB')

    cnb_bar = plt.bar(index + bar_width * 5, CNB_metrics, bar_width, alpha=opacity, color='c', label='CNB')

    bnb_bar = plt.bar(index + bar_width * 6, BNB_metrics, bar_width, alpha=opacity, color='y', label='BNB')

    plt.ylim(0, 1)
    plt.ylabel('Scores')
    plt.title('Metric Scores for Classifications')
    plt.xticks(index + bar_width * 3, metrics)
    plt.legend()

    plt.tight_layout()
    plt.show()
