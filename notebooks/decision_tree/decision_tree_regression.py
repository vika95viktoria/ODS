import numpy as np

def mse(y):
    length = len(y)
    mean_y = sum(y) / length
    return sum([(elem - mean_y)**2 for elem in y]) / length

def regression_var_criterion(X, y, t):
    x_left = X[X < t]
    x_right = X[X >= t]
    d_init = mse(y)
    d_left = mse(y[X < t])
    d_right = mse(y[X >= t])
    return d_init - (len(x_left) / len(X)) * d_left - (len(x_right) / len(X)) * d_right


X = np.linspace(-2, 2, 7)
y = X ** 3  # original dependecy
regression_var_criterion(X, y, 1.3)