# -*- coding: utf-8 -*-

# Classification template



# <------------------  Data Preprocessing ------------------------------->


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv("Artificial_Neural_Networks/ANN build/ChurnData.csv")



#Independent data from credit score to estimated salary
X = dataset.iloc[:,3 : 13].values
# Dependent variable i.e Exited
y = dataset.iloc[:,13].values



# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# Now we have 2 catergorical data at column of country and gender
# index 1 and 2 must be encoded to 0,1,.... data. This is required for training and computation
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_1 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

# Using One Hot Encoding the three categorical features
# france,spain and Germany are encoded into three different random columns
# 0,1,2 will randomly be assigned to these countries
ct = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Now normalize the three columns into 2
X = X[ : , 1 : ]


# Splitting the dataset into the Training set and Test set
# Spliting 80% data into traninng set and 20% into Testing test.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
# this is required to have the same magnitude range of values
# here between -1 and 1
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
# <------------------Making the ANN ------------------------------->

# here the sequiential classifier model is used and Then all
# input hidden and output layers will be applied to that model
import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

# Now we build neural network by applying input ,hidden and output layers

# INPUT AND HIDDEN LAYER
# USE RELU FUNCTION IN INPUT AND HIDDEN LAYER
# FOR OUTPUT WE REQUIRED PROBABLITY OF 2 OUTCOMES SO WE USE SIGMOID AF
# for more than two output layers in classification then softmax activation function is used.

classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# kernal-inintializer means weights close to 0
# units means avg of independent and dependent variables, here 11+1/2==6
# units are hidden layers
# 11 are input independent layers
# 1 is output layer
# AF is relu

classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


classifier.fit(X_train , y_train , batch_size=10 , epochs=100)







# Fitting classifier to the Training set
# Create your classifier here
# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Now any Random Prediction is done by this way, if new data is to be analyzed
 # for new  entry
 new_predict  = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])))
 new_predict = (new_predict > 0.5)



 new_predict1  = classifier.predict(sc.transform(np.array([[0,0,600,1,40,3,6000,2,1,1,5000]])))
 new_predict1 = (new_predict1 > 0.5)








# <---------------------------------------------------------------------------------->
#Evaluating, Improving and Tuning the ANN 
# here k-fold method of scikit_learn is applied to keras model
# for computing the final average accuracy by running it in 10 folds
# So therefore cv is 10 folds and njobs is -1 as all CPU will be run in parallel
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


# Drop Out regularization

# Overfitting :- Model is trained too much on training set that will have less performance over the test set
# it is difference in accuaracy between training and testing data set

                                 
# DropOut is applied to neurons to disable the layers.
# this is the solution to overfitting
# To disable the relations between the inputvariables.


