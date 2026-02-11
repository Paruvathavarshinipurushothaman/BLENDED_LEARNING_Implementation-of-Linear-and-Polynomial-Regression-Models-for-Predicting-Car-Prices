# BLENDED_LEARNING
# Implementation-of-Linear-and-Polynomial-Regression-Models-for-Predicting-Car-Prices

## AIM:
To write a program to predict car prices using Linear Regression and Polynomial Regression models.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset, preprocess the data (handle missing values, encode categorical variables), and split it into training and testing sets.

2. Train a Linear Regression model using the training data and predict car prices for the test data.

3. Transform features into polynomial features, train a Polynomial Regression model, and predict prices.

4. Evaluate and compare both models using metrics like MAE, MSE, and R² score to select the best model.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score 
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('encoded_car_data (1).csv')
print(df.head())

#Select features & target
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]
y = df['price']

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#1. Linear Regression (with scaling)
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
lr.fit(X_train, y_train)
y_pred_linear = lr.predict(X_test)

#2. Polynomial Regression (degree=2)
poly_model = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)

#Evaluate models
print('Name: PARUVATHA VARSHINI P S')
print('Reg. No: 212225100033')
print("Linear Regression:")
mse=mean_squared_error(y_test,y_pred_linear)
print('MSE= ',mean_squared_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)

print('MAE= ',mean_absolute_error(y_test,y_pred_linear))
r2score=r2_score(y_test,y_pred_linear)

print("\nPolynomial Regression:")
print(f"MSE: {mean_squared_error(y_test, y_pred_poly):.2f}")
print(f"R²: {r2_score(y_test, y_pred_poly):.2f}")

#Plot actual vs predicted
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred_linear, label='Linear', alpha=0.6)
plt.scatter(y_test, y_pred_poly, label='Polynomial (degree=2)', alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear vs Polynomial Regression")
plt.legend()
plt.show()
```

## OUTPUT:
<img width="912" height="660" alt="image" src="https://github.com/user-attachments/assets/27611d25-ce7a-4280-bb3e-f1e87796a179" />
<img width="1152" height="595" alt="image" src="https://github.com/user-attachments/assets/8ae77f8f-d736-4fae-96d5-fe56afb00b33" />


## RESULT:
Thus, the program to implement Linear and Polynomial Regression models for predicting car prices was written and verified using Python programming.
