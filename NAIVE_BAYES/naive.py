import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#####DAJE SLABE WYNIKI######################

X = pd.read_csv('datatrain.csv').sample(frac=1).reset_index(drop=True)
Y = X.pop('LABEL').values
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2)

clf = BernoulliNB(alpha=0.5)
clf.fit(X, Y)
BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
Y_pred = clf.predict(X_test)
print("Accuracy: %f" % accuracy_score(Y_test, Y_pred))
print('Classification report')
print(classification_report(Y_test,Y_pred))
