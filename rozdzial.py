#rozdzielanie
import pandas as pd
from sklearn.model_selection import train_test_split

def list_into_dict_train(label, keys):
    dictionary = {}
    dictionary['label'] = label
    
    for i in range(0,len(keys)):
        dictionary['pixel{0}'.format(i)] = str(keys[i])

    return dictionary

def list_into_dict_test(keys):
    dictionary = {}
    
    for i in range(0,len(keys)):
        dictionary['pixel{0}'.format(i)] = str(keys[i])

    return dictionary

def create_index():
    index = []
    index.append('label')    
    index += ['pixel{0}'.format(i) for i in range(0,784)]
    return index

def create_index_test():
    return ['pixel{0}'.format(i) for i in range(0,784)]

def create_series_test(dataa):
    return pd.Series(list_into_dict_test(dataa), TEST_INDEX, dtype=object)

def create_series_train(label,dataa):
    return pd.Series(list_into_dict_train(label, dataa), TRAIN_INDEX, dtype=object)

X = pd.read_csv('obrazy_out.csv')
X = X.sample(frac=1).reset_index(drop=True)

Y = X.pop('label').values
X_train, X_test, Y_train, Y_test = train_test_split(X.values, Y, test_size=0.2)

TEST_INDEX = create_index_test()
TRAIN_INDEX = create_index()

print(Y_train)
print(X_train)

df2 = pd.DataFrame([create_series_test(test) for test in X_test], columns=TEST_INDEX)
df2.to_csv('./obrazy_test.csv')

df3 = pd.DataFrame([create_series_train(Y_train[x],X_train[x]) for x in range(0,len(X_train))], columns=TRAIN_INDEX)
df3.to_csv('./obrazy_train.csv')
