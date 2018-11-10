import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV

X = pd.read_csv('datatrain.csv').sample(frac=1).reset_index(drop=True)
Y = X.pop('LABEL').values
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2)

#Setup arrays to store training and test accuracies
#neighbors = np.arange(1,12)
#train_accuracy =np.empty(len(neighbors))
#test_accuracy = np.empty(len(neighbors))

#for i,k in enumerate(neighbors):
    #Setup a knn classifier with k neighbors
#    knn = KNeighborsClassifier(n_neighbors=k)
    
    #Fit the model
#    knn.fit(X_train, Y_train)
    
    #Compute accuracy on the training set
#    train_accuracy[i] = knn.score(X_train, Y_train)
    
    #Compute accuracy on the test set
#    test_accuracy[i] = knn.score(X_test, Y_test) 

#Generate plot
#plt.title('k-NN Varying number of neighbors')
#plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
#plt.plot(neighbors, train_accuracy, label='Training accuracy')
#plt.legend()
#plt.xlabel('Number of neighbors')
#plt.ylabel('Accuracy')
#plt.show()

#best results for max 2 neighbours

#k parameter tuning
print('k-parameter tuning...')
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=20)
knn_cv.fit(X,Y)
print('Best achieved score: %f' % knn_cv.best_score_)
print('k-parameter value: %i' % knn_cv.best_params_['n_neighbors'])

#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier(n_neighbors=knn_cv.best_params_['n_neighbors'])
knn.fit(X_train,Y_train)
print('Accuracy %f' % knn.score(X_test,Y_test))

Y_pred = knn.predict(X_test)
print('Classification report')
print(classification_report(Y_test,Y_pred))
