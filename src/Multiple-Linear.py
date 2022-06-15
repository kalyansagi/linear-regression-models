# Importing libraries
# -------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# ---------------------
dataset = pd.read_csv('Multiple-Linear-Dataset.csv')
X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, 4].values
print(X)
print(y)

# Encoding categorical data
# -------------------------
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# import numpy as np
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
# X = np.array(ct.fit_transform(X))

# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# encoder = OneHotEncoder(drop='first', dtype=int)
# ct = ColumnTransformer([('categorical_encoding', encoder, [3])], remainder='passthrough')
# X = ct.fit_transform(X)

# Avoiding the Dummy Variable Trap
# X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# --------------------------------------------------------
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_predict = regressor.predict(X_test)

# comparing the scores of test & training sets to verify the accuracy of model.
print("Model score on Testing data", regressor.score(X_test, y_test))
print("Model score on Training data", regressor.score(X_train, y_train))


# Display the predicted vs actual values to verify the accuracy of model.
df = pd.DataFrame(data={'Predicted value': y_predict.flatten(), 'Actual Value': y_test.flatten()})
print(df)

# RMSE is also another factor to indicate the accuracy of model. See below for the r2 score for my model.
from sklearn.metrics import r2_score
score = r2_score(y_test, y_predict)
print("RMSE is", score)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50, 1)).astype(int), values=X, axis=1)
X_opt = X[:, [0, 1, 2]]
print(X_opt)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print("regressor_OLS.summary() with 2 features(Product_1 & Product_2)")
print(regressor_OLS.summary())
X_opt = X[:, [0, 1]]
print(X_opt)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print("regressor_OLS.summary() with 1 features(Product_1)")
print(regressor_OLS.summary())

# X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
# X_opt = np.array(X[:, [0, 3]], dtype=float)
# regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
# print(regressor_OLS.summary())
