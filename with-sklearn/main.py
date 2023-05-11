import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split

# 1. Read csv file and shuffle the rows
df = pd.read_csv('diabetes.csv', sep=',').sample(frac=1, random_state=42).reset_index(drop=True)

# 2. Clean the data

#Drop rows where Glucose, Blood Pressure, Skin Thickness, Insulin or BMI is 0, as they are likely not valid data
column_names = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df.drop(df[(df[column_names] == 0).any(axis=1)].index, inplace=True)

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

print(np.shape(X), np.shape(y)) # (392, 8) and (392,)

#Split data into train and test sets with a 4:1 ratio
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Define range of K neighbors to try training with
K_ranges = range(1,11)

#Create a gridsearch instance, provide it with the K range
gridsearch = GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': K_ranges})

#Fit the data
gridsearch.fit(x_train, y_train)

#Print the best accuracy score with the best params
print(gridsearch.best_score_, gridsearch.best_params_)

#Predict with the test data
y_pred = gridsearch.predict(x_test)

#Plot the predicted data onto a confusion matrix
sns.heatmap(confusion_matrix(y_test, y_pred))
plt.show()