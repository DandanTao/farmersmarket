import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def analyze(SGD_metrics, LSVC_metrics, LR_metrics, CNB_metrics):
    n_groups = 4
    metrics = ('Accuracy', 'F1', 'Recall', 'Precision')

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.15
    opacity = 0.8

    sgd_bar = plt.bar(index, SGD_metrics, bar_width, alpha=opacity, color='b', label='SGD')
    lsvc_bar = plt.bar(index + bar_width, LSVC_metrics, bar_width, alpha=opacity, color='k', label='LSVC')
    lr_bar = plt.bar(index + bar_width * 2, LR_metrics, bar_width, alpha=opacity, color='m', label='LR')
    cnb_bar = plt.bar(index + bar_width * 3, CNB_metrics, bar_width, alpha=opacity, color='c', label='CNB')

    plt.ylim(0, 1)
    plt.ylabel('Scores')
    plt.title('Metric Scores for Classifications')
    plt.xticks(index + bar_width, metrics)
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    Modified code from SKlearn
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
