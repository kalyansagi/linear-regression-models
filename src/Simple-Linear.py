# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Simple-Linear-Dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


# Splitting the dataset into the Training set and Test set
# --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Feature scaling
# ---------------
# from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)

print(X_train, X_test, y_train, y_test)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)

# Predicting the Test set results
y_predict = linear_regression.predict(X_test)

#Find out the RMSE, expected to be in 0-1 range. This is optional;
from sklearn.metrics import r2_score
score = r2_score(y_test, y_predict)
print("r2 score for the model is", score)

# Visualising the Training set results
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, linear_regression.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color='red')
plt.plot(X_train, linear_regression.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
