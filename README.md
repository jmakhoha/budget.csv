import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv("budget.csv")

df.head()

df.tail()

# Calculate the correlation coefficients between Revenue and other variables
correlation = df.corr()["Revenue ($M)"].drop("Revenue ($M)")
strongly_related_variables = correlation[abs(correlation) >= 0.5].index.tolist()
print("Variables strongly related to Revenue ($M):", strongly_related_variables)

# Prepare the data
x = df[["Global GDP Index"]]
y = df["Revenue ($M)"]

# Create and fit the model
model = LinearRegression()
model.fit(x, y)

# Plotting
plt.scatter(x, y, color='blue')
plt.plot(x, model.predict(x), color='red')
plt.title('Global GDP Index vs Revenue')
plt.xlabel('Global GDP Index')
plt.ylabel('Revenue ($M)')
plt.show()

# Model coefficients
intercept = model.intercept_
coefficient = model.coef_[0]
print("Intercept:", intercept)
print("Coefficient:", coefficient)
print("Model relationship: y = {:.2f}x + {:.2f}".format(coefficient, intercept))

#Q2: Multiple Linear Regression
X_multi = df[['Global GDP Index', "Cust Serv Calls ('000s)", "Employees ('000)", "Items ('000)"]]
y_multi = df['Revenue ($M)']

# Fit the multiple linear regression model
model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)

# Get the intercept and coefficients
intercept_multi = model_multi.intercept_
coefficients_multi = model_multi.coef_

print("Intercept (Multiple Linear Regression):", intercept_multi)
print("Coefficients (Multiple Linear Regression):", coefficients_multi)
print("Model relationship: Y = {} + {}X_1 + {}X_2 + {}X_3 + {}X_4".format(intercept_multi, coefficients_multi[0], coefficients_multi[1], coefficients_multi[2], coefficients_multi[3]))







