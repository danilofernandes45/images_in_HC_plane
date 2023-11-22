import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

folds = 10
kf = KFold(n_splits=folds)
data = pd.read_csv("BP_data.csv")
label = data["Label"]
data.drop(columns = ["Label", "img_id"], inplace = True)
label_summary = label.unique()
label = label.apply(lambda x : np.where(label_summary == x)[0][0])

all_acc = []
all_recall = []
all_precision = []
for i in range(100):
    acc = []
    recall = []
    precision = []
    data = data.sample(frac = 1, random_state = i)
    label = label.sample(frac = 1, random_state = i)
    for indexes in kf.split(data):
        X_train = data.iloc[indexes[0]]
        X_test = data.iloc[indexes[1]]

        y_train = label.iloc[indexes[0]]
        y_test = label.iloc[indexes[1]]
        
        classifier = KNeighborsClassifier(n_neighbors = 5)
        # classifier = svm.SVC()
        model = classifier.fit(X_train ,y_train)
        y_pred = model.predict(X_test)

        acc.append(accuracy_score(y_test,y_pred))
        recall.append(recall_score(y_test,y_pred, average = "macro"))
        precision.append(precision_score(y_test,y_pred, average = "macro"))
    
    all_acc.append(np.mean(acc))
    all_recall.append(np.mean(recall))
    all_precision.append(np.mean(precision))

print("Accuracy: %f"%np.mean(all_acc))
print("Recall: %f"%np.mean(all_recall))
print("Precision: %f"%np.mean(all_precision))