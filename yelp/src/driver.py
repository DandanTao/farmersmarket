import RB_classification
import ML_classification

PATH1="../data/yelp_labelling_1000.csv"
PATH2="../data/1000_more_yelp.csv"
PATH3="../data/2000_yelp_labeled.csv"

def draw_table(r1, r2, r3, r4, r5, r6):
    import matplotlib.pyplot as plt
    bow, tfidf = RB_classification.run_all(iter, PATH3)
    sgd, lsvc, lr, cnb = ML_classification.run_all(cross_val=iter,
            analyze_metrics=True,
            confusion_matrix=True,
            file_path=PATH3)
