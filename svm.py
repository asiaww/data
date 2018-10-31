from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn import svm

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

i = 0
acc = []
pr = []
rec = []
f1 = []


while i < 100:
    X = pd.read_csv('datatrain.csv')
    X = X.sample(frac=1).reset_index(drop=True)

    Y = X.pop('LABEL').values
    X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2)

    clf = svm.SVC(kernel='rbf', gamma=0.1, C=100.)
    #print('Training...')
    clf.fit(X_train, Y_train)

    Y_result = clf.predict(X_test)
    acc.append(accuracy_score(Y_test, Y_result))    
    #print('Accuracy: %.2f' % accuracy_score(Y_test, Y_result))

    PRF = precision_recall_fscore_support(Y_test, Y_result, average='macro')
    pr.append(PRF[0])
    rec.append(PRF[1])
    f1.append(PRF[2])
    
    #print('Precision: %.2f' % PRF[0])
    #print('Recall: %.2f' % PRF[1])
    #print('F1: %.2f' % PRF[2])
    
    i+= 1

print(sum(acc) / float(len(acc)))
print(sum(pr) / float(len(pr)))
print(sum(rec) / float(len(rec)))
print(sum(f1) / float(len(f1)))
