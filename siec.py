# import libraries
import numpy as np # linear algebra, matrix multiplications
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import sqrt
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

from keras import backend as K
import datetime

# for the architecture
from keras.models import Sequential
from keras.layers import Dense, Dropout, Lambda, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPool2D, AvgPool2D

# optimizer, data generator and learning rate reductor
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# reshape of image data to (nimg, img_rows, img_cols, 1)
def df_reshape(df):
    df = df.values.reshape(-1, dim, dim, 1) 
    # -1 means the dimension doesn't change, so 42000 in the case of xtrain and 28000 in the case of test
    return df

if __name__ == '__main__':

    train = pd.read_csv("obrazy_train.csv")
    test = pd.read_csv("obrazy_test.csv")

    ntrain = train.shape[0]
    #ntest = test.shape[0]

    # array containing labels of each image
    ytrain = train["label"]

    #dataframe containing all pixels (the label column is dropped)
    xtrain = train.drop("label", axis=1)

    #X = pd.read_csv("obrazy_out.csv") 
    #X = X.sample(frac=1).reset_index(drop=True)

    #Y = X.pop('label').values
    #xtrain, test, ytrain, ytest = train_test_split(X.values, Y, test_size=0.1)   

    # the images are in square form, so dim*dim = 784
    dim = int(sqrt(xtrain.shape[1]))

    # Normalize the data
    xtrain = xtrain / 255.0
    test = test / 255.0

    xtrain = xtrain.fillna(1.0)
    test = test.fillna(1.0)

    xtrain = df_reshape(xtrain) # numpy.ndarray type
    test = df_reshape(test) # numpy.ndarray type

    # number of classes, in this case 10
    nclasses = ytrain.max() - ytrain.min() + 1
    ytrain = to_categorical(ytrain, num_classes = nclasses)

    # fix random seed for reproducibility
    seed = 2
    np.random.seed(seed)

    # percentage of xtrain which will be xval
    split_pct = 0.1

    # Split the train and the validation set
    xtrain, xval, ytrain, yval = train_test_split(xtrain,ytrain, test_size=split_pct,random_state=seed,shuffle=True,stratify=ytrain)
    print(xtrain.shape, ytrain.shape, xval.shape, yval.shape)

    model = Sequential()

    dim = 28
    nclasses = 2

    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu', input_shape=(dim,dim,1)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='same', activation='relu',))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.9))

    #model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
    #model.add(Conv2D(filters=64, kernel_size=(5,5), padding='same', activation='relu'))
    #model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    #model.add(Dropout(0.9))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(nclasses, activation='softmax'))
    
    print(model.summary())

    datagen = ImageDataGenerator(
          featurewise_center=False,            # set input mean to 0 over the dataset
          samplewise_center=False,             # set each sample mean to 0
          featurewise_std_normalization=False, # divide inputs by std of the dataset
          samplewise_std_normalization=False,  # divide each input by its std
          zca_whitening=False,                 # apply ZCA whitening
          rotation_range=0,                   # randomly rotate images in the range (degrees, 0 to 180)
          zoom_range =0,                    # Randomly zoom image 
          width_shift_range=0,               # randomly shift images horizontally (fraction of total width)
          height_shift_range=0,              # randomly shift images vertically (fraction of total height)
          horizontal_flip=False,               # randomly flip images
          vertical_flip=False)                 # randomly flip images

    datagen.fit(xtrain)

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

    epochs = 15
    batch_size = 64

    print('Start')
    print(datetime.datetime.now())
    history = model.fit_generator(datagen.flow(xtrain,ytrain, batch_size=batch_size),
                              epochs=epochs, 
                              validation_data=(xval,yval),
                              verbose=1, 
                              steps_per_epoch=xtrain.shape[0] // batch_size, 
                              callbacks=[])

    print('End')
    print(datetime.datetime.now())
