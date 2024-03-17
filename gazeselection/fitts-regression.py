'''
This file holds the code to fit the Fitts' law curve and find the regression coefficients.
'''
import numpy as np
from scipy.optimize import curve_fit

# Given data
A = 10  # degrees
W = np.array([1, 1.5, 2, 3, 4, 5])  # degrees
MT = np.array([0.520, 0.443, 0.367, 0.310, 0.257, 0.257])  # seconds

# Fitt's Law equation
def fitts_law(W, a, b):
    return a + (b * np.log2(2*A / W))


#Fit the data using curve_fit
params, covariance = curve_fit(fitts_law, W, MT)

# Extract coefficients
a = params[0]
b = params[1]

print("Coefficient a:", a)
print("Coefficient b:", b)

#Answer
# a = 2.86
# b = 147.46
