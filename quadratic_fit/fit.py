import time
import io, os, sys
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

file_path = r"/lus/eagle/projects/RECUP/twang/physics-theia/systematic_variation.dat"

#================Loading data, no normalization================#

x1, x2, y = [], [], []
with open(file_path, 'r') as f:
    for line in f:
        a, b, val = map(float, line.split())
        x1.append(a)
        x2.append(b)
        y.append(val)

x1 = np.array(x1)
x2 = np.array(x2)
y = np.array(y)


def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

######=============fit y vs x2============
for x1_index in np.arange(0, 100, 1):
    index = np.arange(x1_index * 100, x1_index * 100 + 100, 1)
    x2_fit = x2[index]
    y_fit = y[index]
#    print(x1[index])
#    print(x2_fit)
#    print(y_fit)
    params, covariance = curve_fit(quadratic_function, x2_fit, y_fit)
    a, b, c = params
    aerr, berr, cerr = np.sqrt(np.diag(covariance))
    print(f'x1 = {x1[int(x1_index * 100)]}, a = {a} +- {aerr}, b = {b} +- {berr}, c = {c} +- {cerr}')

######=============fit y vs x1============
#for x2_index in np.arange(0, 100, 1):
#    index = np.arange(x2_index, x2_index + 10000, 100)
#    x1_fit = x1[index]
#    y_fit = y[index]
##    print(x1_fit)
##    print(x2[index])
##    print(y_fit)
#    params, covariance = curve_fit(quadratic_function, x1_fit, y_fit)
#    a, b, c = params
#    aerr, berr, cerr = np.sqrt(np.diag(covariance))
#    print(f'x2 = {x2[x2_index]}, a = {a} +- {aerr}, b = {b} +- {berr}, c = {c} +- {cerr}')

######=============fit y vs x1 and x2========
X = np.column_stack((x1, x2))
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
coef = model.coef_
intercept = model.intercept_
y_pred = model.predict(X_poly)
diff = y - y_pred

print("Intercept:", intercept)
print("Coefficients:", coef)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(x1, x2, c=diff, cmap='viridis', marker='o')
cbar = plt.colorbar(scatter)
cbar.set_label('Difference')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('2D Color Plot of Differences')

plt.savefig('diff_plot.png')
