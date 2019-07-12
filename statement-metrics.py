from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, precision_recall_curve, roc_auc_score
import pandas as pd
import numpy

data = pd.read_csv('classification.csv')
y_true = data.iloc[:, 0]
y_pred = data.iloc[:, 1]

tp = fp = fn = tn = 0

for i in range(len(y_true)):
    if y_true[i]:
        if y_pred[i]:
            tp += 1
        else:
            fn += 1
    else:
        if y_pred[i]:
            fp += 1
        else:
            tn += 1
first_answer = [tp, fp, fn, tn]
print(first_answer)
second_answer = '{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}' .format( accuracy_score(y_true, y_pred),  precision_score(y_true, y_pred),recall_score(y_true, y_pred),f1_score(y_true, y_pred))
print(second_answer)

data_scores = pd.read_csv('scores.csv')

y2_true, score_logreg, score_svm, score_knn, score_tree = data_scores.iloc[:, 0], data_scores.iloc[:, 1], data_scores.iloc[:, 2], data_scores.iloc[:, 3], data_scores.iloc[:, 4]

third_answer = [roc_auc_score(y2_true, score_logreg),
                roc_auc_score(y2_true, score_svm),
                roc_auc_score(y2_true, score_knn),
                roc_auc_score(y2_true, score_tree),]
print(third_answer)
print('score_logreg')

def max_precision_with_70_recall(y_true, y_test):
    precision_max = []
    prec_array, recall_array, _ = precision_recall_curve(y_true, y_test)

    for i in range(len(prec_array)):
        if recall_array[i] > 0.7:
            precision_max.append(recall_array[i])
    return numpy.mean(precision_max)


fourth_answer = [max_precision_with_70_recall(y_true, score_logreg),
                 max_precision_with_70_recall(y_true, score_svm),
                 max_precision_with_70_recall(y_true, score_knn),
                 max_precision_with_70_recall(y_true, score_tree),

                 ]
print(fourth_answer)
