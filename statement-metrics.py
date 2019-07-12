from sklearn.metrics import accuracy_score, precision_score, recall_score, \
    f1_score, precision_recall_curve
import pandas as pd

data = pd.read_csv('classification.csv')
y_true = data.iloc[:,0]
y_pred = data.iloc[:,1]

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
second_answer = [accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred)]
print(second_answer)

data2 = pd.read_csv('scores.csv')
y2_true = data2.iloc[:,0]