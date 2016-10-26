import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast_cancer_data.txt')

#replace missind data with -99999, drop unnecessay columns.
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

#set input and output variables as numpy array.
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

#cross validation to get train test data for testing accuracy.
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

#using Knearest neighbour classification model.
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print('accuracy', accuracy)

#examples to predict
example_measures = np.array([[8,8,10,8,7,7,9,7,1], [1,3,3,4,4,5,4,7,1]])
prediction = clf.predict(example_measures)
print('prediction', prediction)