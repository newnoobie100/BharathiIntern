# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming you have a CSV file named 'house_data.csv')
data = pd.read_csv('house_data.csv')

# Assuming 'X' contains features and 'y' contains target prices
# Replace with actual feature columns
X = data[['feature1', 'feature2', 'feature3']]
y = data['price']  # Replace with actual target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Squared Error to evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Visualize the predicted vs. actual prices (for simplicity, let's assume a single feature)
plt.scatter(X_test['feature1'], y_test, color='blue', label='Actual')
plt.plot(X_test['feature1'], y_pred, color='red', label='Predicted')
plt.xlabel('Feature 1')
plt.ylabel('Price')
plt.legend()
plt.show()
