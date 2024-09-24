import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from pylab import rcParams

rcParams['figure.figsize'] = 14, 8
RANDOM_SEED = 42
LABELS = ['Normal', 'Fraud']

#Reading the data
df = pd.read_csv('creditcard.csv')

#Getting the info of dataset
#print(df.info())

#Data Visualisation(remove quotes to run and see)
'''
classes = pd.Series(df['Class']).value_counts()
classes.plot(kind = 'bar', rot = 0)
plt.title("Credit Card Transactions")
plt.xticks(range(2), LABELS)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.show()
'''

#Dimensions of normal and fraud data
normal = df[df['Class'] == 0]
fraud = df[df['Class'] == 1]
print( normal.shape, fraud.shape)

#If u want more information about fraud data with respective to amount
#print(fraud.Amount.describe())

#To know how many outliers are there
outlier_fraction = len(fraud)/float(len(normal))
print(outlier_fraction)
print(f'Normal : {len(normal)}')
print(f'Normal : {len(fraud)}')

#Create independent and Dependent Features
columns = df.columns.tolist()
#Filter the columns to remove data we do not want
columns = [c for c in columns if c not in ['Class']]
#Store teh variable we are predicting
target = 'Class'
#Define a random state
state = np.random.RandomState(42)
X = df[columns]
y = df[target]
X_outliers = state.uniform(low = 0, high = 1, size = (X.shape[0], X.shape[1]))
#Print the shape of X and Y
print(X.shape)
print(y.shape)

#Model Creation
#Isolation Forests algorithm
model = IsolationForest(n_estimators = 100, max_samples = len(X),
                        contamination = outlier_fraction, random_state = state,
                        verbose = 0)

model.fit(X)
scores_prediction = model.decision_function(X)
y_predict = model.predict(X)

#Reshape the prediction values
y_predict[y_predict == 1] = 0
y_predict[y_predict == -1] = 1
n_errors = (y_predict != y).sum()

#Classification Metrics
print(n_errors)
print('Accuracy Score : ', accuracy_score(y, y_predict))
print("Classification Report:",classification_report(y, y_predict))



