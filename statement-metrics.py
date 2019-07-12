from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

data = pd.read_csv('classification.csv')
y_true = data.iloc[:,0]
y_pred = data.iloc[:,1]

first_answer = [accuracy_score(y_true, y_pred),
                precision_score(y_true, y_pred),
                recall_score(y_true, y_pred),
                f1_score(y_true, y_pred)]
print(first_answer)

data = 