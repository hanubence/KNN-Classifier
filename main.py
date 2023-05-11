import seaborn as sns
import matplotlib.pyplot as plt
from KNN import KNNClassifier

kn = KNNClassifier(5, 0.1)

#Load dataset and split the data into train and test
x,y = kn.load_csv('diabetes.csv')
kn.train_test_split(x,y)

#Predict using our test dataset
kn.predict(kn.x_test)

#Print model accuracy
print(kn.accuracy())

#Find and print best parameters using grid-search
print(kn.best_k())

sns.heatmap(kn.confusion_matrix())
plt.show()