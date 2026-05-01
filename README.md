# SGD-Regressor-for-Multivariate-Linear-Regression
# SINDHU PRIYA REDDY G
# 212224040319
## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize weights and bias with small random values, and choose a learning rate.

2. For each training example, predict output using the multivariate linear equation.

4.  Compute the error and update weights and bias using SGD (gradient of loss function).

5.  Repeat for multiple epochs until the model converges or error is minimized.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: DHANUSH C
RegisterNumber:212224040066
*/


import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

data = fetch_california_housing()

X = data.data[:, :3]
Y = np.c_[data.target, data.data[:, 6]]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MultiOutputRegressor(SGDRegressor(random_state=42))
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nSample Predictions (House Price, Population):")
print(Y_pred[:5])


```



## Output:


<img width="673" height="190" alt="image" src="https://github.com/user-attachments/assets/3481bb73-a7bd-4575-9326-5f11335ead51" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
