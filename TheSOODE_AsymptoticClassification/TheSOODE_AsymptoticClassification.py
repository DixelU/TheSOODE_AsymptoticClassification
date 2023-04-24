import numpy as np
import math
import warnings
import random
import re
from numpy.core.defchararray import asarray

from numpy.lib.function_base import append
from tqdm import tqdm
from sklearn.neural_network import MLPRegressor
from matplotlib.figure import Figure
from scipy.integrate import RK45
from matplotlib import pyplot as plt

WITH_ODE_SOLUTION_PREDICT = False

fig: Figure = plt.figure()

def f(X, p):
    return np.array([
           p[0]*X[0] + p[1]*X[1],
           p[2]*X[0] + p[3]*X[1]
        ], dtype=np.float64)

def createFunc(p):
    return lambda t, y: f(y, p) 

def getSolution(rk, iters):
    line = [rk.y]
    for i in range(1, iters):
        res = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = rk.step()
        if res != None or rk.status != "running":
            break
        line.append(rk.y)
    return line


xmin = -5
ymin = -5
xmax = 5
ymax = 5

p = [random.uniform(-5, 5),
    random.uniform(-5, 5),
    random.uniform(-5, 5),
    random.uniform(-5, 5)]

N = 100
RK_iters = 100
T_bound = 100
func = createFunc(p)
solutions = []

for i in range(N):
    x_0 = random.uniform(xmin, xmax)
    y_0 = random.uniform(ymin, ymax)
    rk = RK45(func, 0, np.array([x_0, y_0], dtype=np.float64), T_bound)
    new_solution = getSolution(rk, RK_iters)
    if np.isnan(new_solution).any():
        continue
    solutions.append(new_solution)
solutions = np.array(solutions)

# is a primitive solution's embedding extractor;
def polarAngleExtractor(solutions):
    last_points = solutions[:, -1];
    complex_array = last_points[:, 0] + 1j * last_points[:, 1]; 
    return np.angle(complex_array);

# takes an array of 2xN arrays (paths)
def drawSolutions(solutions):
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    for i in range(len(solutions)):
        plt.scatter(solutions[i][:, 0],solutions[i][:, 1], marker='x')
        plt.plot(solutions[i][:, 0], solutions[i][:, 1], linestyle="-.", linewidth=1)    
    plt.show()

# returns "classes" of solution
def groundTruthClassifier(solutions):

print(polarAngleExtractor(solutions))
drawSolutions(solutions)