import itertools
import numpy as np
import math
import pylab as pl
import warnings
import random
import re
import colorsys
import seaborn as sns

from tqdm import tqdm
from matplotlib.figure import Figure
from scipy.integrate import RK45, RK23, DOP853
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

from numba import njit

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

# is a primitive solution's embedding extractor
def polarAngleExtractor(solutions):
    last_points = solutions[:, -1]
    complex_array = last_points[:, 0] + 1j * last_points[:, 1] 
    return np.angle(complex_array)

# takes an array of 2xN arrays (paths)
# bg = (points, colors)
def drawSolutions(solutions, xrange, yrange, bg_colors=None):
    plt.xlim(xrange[0], xrange[1])
    plt.ylim(yrange[0], yrange[1])
    if bg_colors:
        x = bg_colors['points'][:,0]
        y = bg_colors['points'][:,1]
        maximal_confidence = bg_colors['m_conf']
        colors = bg_colors['colors']
        
        #plt.scatter(x, y, (1. - np.array(maximal_confidence))*50, c=colors)

        ax = plt.gca()
        #center = ax.tricontourf(x, y, maximal_confidence, levels=20, linewidths=0.5, colors="k")
        center = ax.tricontourf(x, y, maximal_confidence, levels=20, cmap="RdBu_r")
        fig.colorbar(center, ax=ax)

    for i in range(len(solutions)):
        plt.scatter(solutions[i][0][0],solutions[i][0][1], marker='x')
        plt.plot(solutions[i][:, 0], solutions[i][:, 1], linestyle=":", linewidth=1)
    plt.show()

def __GT_base1_classifier(solutions, parameters):
    classes = []

    a = parameters[0]
    b = parameters[1]
    c = parameters[2]
    d = parameters[3]

    l1 = ((a + d) + ((a + d) ** 2 - 4 * (a * d - b * c)) ** 0.5) * 0.5
    l2 = ((a + d) - ((a + d) ** 2 - 4 * (a * d - b * c)) ** 0.5) * 0.5

    is_complex = isinstance(l1, complex) or isinstance(l2, complex)

    diverging_knot = (not is_complex) and l1.real > 0 and l2.real > 0
    converging_knot = (not is_complex) and l1.real < 0 and l2.real < 0
    saddle = (not is_complex) and l1.real * l2.real < 0

    constant_field = (not is_complex) and (l1.real == 0 or l2.real == 0)

    converging_focus = (is_complex) and l1.real < 0
    diverging_focus = (is_complex) and l1.real > 0
    stationary_focus = (is_complex) and l1.real == 0

    detected_type = "unknown"
    if diverging_knot:
        detected_type = "diverging knot"
    if converging_knot:
        detected_type = "converging knot"
    if saddle:
        detected_type = "saddle"
    if constant_field:
        detected_type = "constant_field"
    if converging_focus:
        detected_type = "converging focus"
    if diverging_focus:
        detected_type = "diverging focus"
    if stationary_focus:
        detected_type = "stationary focus"

    x1 = 1
    x2 = 1
    y1 = (l1 - a) / b
    y2 = (l2 - a) / b

    for solution in solutions:
        x0 = solution[0][0]
        y0 = solution[1][1] 

        angle0 = (np.angle(x0 + 1j*y0) + 2 * math.pi) % (math.pi * 2)
        angle1 = (np.angle(x1 + 1j*y1) + 2 * math.pi) % (math.pi * 2)
        angle2 = (np.angle(x2 + 1j*y2) + 2 * math.pi) % (math.pi * 2)

        angle1_0 = angle1 % math.pi 
        angle1_1 = angle1_0 + math.pi
        angle2_0 = angle2 % math.pi
        angle2_1 = angle2_0 + math.pi

        if angle1_0 > angle2_0:
            angle1_0, angle1_1, angle2_0, angle2_1 = angle2_0, angle2_1, angle1_0, angle1_1  

        zone00 = angle1_0 < angle0 and angle0 <= angle2_0 
        zone01 = angle2_0 < angle0 and angle0 <= angle1_1
        zone10 = angle1_1 < angle0 and angle0 <= angle2_1 
        zone11 = (angle2_1 < angle0 and angle0 <= 2*math.pi) or angle0 < angle1_0

        classes.append((zone00, zone01, zone10, zone11))
    return detected_type, classes

def makeSomeSolutions(N, dims, initvals_range):
    solutions = []

    for i in range(N):
        x_i = [random.uniform(initvals_range[i][0], initvals_range[i][1]) for i in range(dims)]

        rk = RK23(func, 0, np.array(x_i, dtype=np.float64), T_bound, max_step=max_step)

        new_solution = getSolution(rk, RK_iters)
        if np.isnan(new_solution).any():
            continue
        solutions.append(new_solution)
    return np.array(solutions)

def runClassificationTrain(initvals_embeddings, classes, pretrained_classifier=None, classifier_name=None, params=None):
    if not pretrained_classifier:
        if classifier_name == "knn":
            pretrained_classifier = \
                KNeighborsClassifier(n_neighbors=params["k"],
                                     n_jobs=params["cores_count"])
    pretrained_classifier.fit(initvals_embeddings, classes)
    return pretrained_classifier

def makeLinearCombinations(points, repeats):
    points = [np.array(_) for _ in points]
    for _ in range(repeats):
        i = int(random.uniform(0, len(points)))
        j = int(random.uniform(0, len(points)))
        t = random.uniform(0, 1)
        points.append((np.array(points[i])*t + (1. - t)*np.array(points[j])))
    return np.array(points)

def makeCubePointsFromRanges(ranges):
    ranges = np.array(ranges)
    
    size = 10
    t_values = [float(i)/size for i in range(size + 1)]
    
    ranges_combination = [ranges[:,0] * t + (1. - t) * ranges[:,1] for t in t_values]
    return list(itertools.product(*zip(*ranges_combination)))

def normalize(points, ranges):
    singlepoint_normalise = lambda p: [(p[i] - ranges[i][0])/(ranges[i][1] - ranges[i][0]) for i in range(len(ranges))]
    return np.array([singlepoint_normalise(pt) for pt in points])

def denormalize(points, ranges):
    singlepoint_normalise = lambda p: [p[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0] for i in range(len(ranges))]
    return np.array([singlepoint_normalise(pt) for pt in points])

def classesProbabilitiesToSingularValues(class_probabilities):
    number_of_classes = len(class_probabilities[0])
    hue_values = np.linspace(0, 1, number_of_classes)
    colors = []
    max_probs = []
    
    for single_element in class_probabilities:
        normalised_probabilites = single_element / np.sum(single_element)
        current_color = np.array([0,0,0], dtype=np.float64)
        for i in range(len(normalised_probabilites)):
            current_color += \
                np.array(colorsys.hsv_to_rgb(
                    hue_values[i],
                   normalised_probabilites[i], 1))\
                        * normalised_probabilites[i]
        colors.append(current_color) 
        max_probs.append(np.max(normalised_probabilites))
    return np.array(colors), np.array(max_probs)

fig: Figure = plt.figure()

SOODE_parameters = [
    random.uniform(-5, 5),
    random.uniform(-5, 5),
    random.uniform(-5, 5),
    random.uniform(-5, 5)]

N = 500
RK_iters = 25
T_bound = 100
max_step = 1
func = createFunc(SOODE_parameters)

initvals_range = list([(-20,20)]*2)
solutions = makeSomeSolutions(N, 2, initvals_range)
detected_type, classes = __GT_base1_classifier(solutions, SOODE_parameters)
numpyfied_classes = np.asarray(np.array(classes) > 0).nonzero()[1]
initval_embeddings = normalize(solutions[:,0], initvals_range)
base_trained_classifier = runClassificationTrain(initval_embeddings, numpyfied_classes, classifier_name="knn", params={"k": 10, "cores_count": 1})

initvals_cube_points = makeCubePointsFromRanges(initvals_range)
some_random_set_of_points_in_cube = makeLinearCombinations(initvals_cube_points, 10000)
normalised_points_in_cube = normalize(some_random_set_of_points_in_cube, initvals_range)

probs = base_trained_classifier.predict_proba(normalised_points_in_cube)
colors,max_confidence  = classesProbabilitiesToSingularValues(probs)

print(detected_type)
drawSolutions(solutions,
              *initvals_range[:2],
              bg_colors={
                  'points': some_random_set_of_points_in_cube, 
                  'm_conf': max_confidence,
                  'colors': colors
              })