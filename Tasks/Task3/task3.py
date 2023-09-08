import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
# Use only the first two features (sepal length and sepal width)
y = iris.target

# Creating a DataFrame for  visualization
iris_df = pd.DataFrame(data=np.c_[
                       iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
# Spliting the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# Creating a KNN classifier with, for example, k=3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)

# Fitting the classifier to the training data
knn.fit(X_train, y_train)
# Making predictions on the test data
y_pred = knn.predict(X_test)
# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Printing a classification report and confusion matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
