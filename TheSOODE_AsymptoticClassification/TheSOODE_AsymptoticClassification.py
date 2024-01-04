from cmath import inf, isinf, nan, pi
import cmath
from copy import deepcopy
from enum import unique
import itertools
import time
import numpy as np
import math
import pylab as pl
import warnings
import random
import re
import joblib
import colorsys
import matplotlib
import seaborn as sns

from tqdm import tqdm
from matplotlib.figure import Figure
from scipy.integrate import RK45, RK23, DOP853, Radau
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from numpy.polynomial.polynomial import polyval, polyfromroots

#from numba import njit
matplotlib.use('Agg')
random.seed(42)

def f_regular(X, p):
    return np.array([
           p[0]*X[0] + p[1]*X[1],
           p[2]*X[0] + p[3]*X[1]
        ], dtype=np.float64)

def f_polynomial(X, p):
    return np.array([polyval(X[0], p), -X[1]], dtype=np.float64)

def f_target(X, p):
    return np.array([
        X[0] * (-p[0] * (X[0] - X[1]) * X[0] * X[2] + 2 * X[1]),
        (X[0] - X[1]) * X[2] * (X[0] ** 2 - 1.),
        X[1] * X[2] * (X[0] ** 2 - 1.)
    ], dtype=np.float64)

def f_pendulum_raw_0_it(X, p, index_begin): #will this work?
    m = p[0]
    l = p[1]
    g = p[2]
    sdml = 6. / (m * l ** 2)
    nhmls = -0.5 * m * l ** 2
    cosdiff = np.cos(X[0 + index_begin] - X[1 + index_begin])
    sindiff = np.sin(X[0 + index_begin] - X[1 + index_begin])
    td1 = sdml * (2 * X[2 + index_begin] - 3 * cosdiff * X[3 + index_begin]) / (16 - 9 * cosdiff ** 2)
    td2 = sdml * (8 * X[3 + index_begin] - 3 * cosdiff * X[2 + index_begin]) / (16 - 9 * cosdiff ** 2)
    ptd1 = nhmls * (td1 * td2 * sindiff + 3 * g * np.sin(X[0 + index_begin]) / l)
    ptd2 = nhmls * (-td1 * td2 * sindiff + g * np.sin(X[1 + index_begin]) / l)
    return np.array([ td1, td2, ptd1, ptd2 ], dtype=np.float64)

def f_two_phase_fluid_in_pipe(X, p):
    v = X[0]
    theta = X[1]
    m = X[2]

    gamma = p[0]
    kappa = p[1]
    s = p[2]
    alpha = p[3]

    F1 = (theta - 1) * (v ** 2) - theta * (v - 1) * (gamma * m * (v - 1) + m + (v ** 2) * (m - 1) / kappa) * alpha
    F2 = - (theta - 1) * (v ** 2) * (gamma * m - 1 + s * theta * (m - 1) / kappa) + theta * (gamma - 1) * m * (v - 1) * (gamma * m * (v - 1) + 1) * alpha
    F3 = (theta - 1) * (v ** 2) * (gamma * m + 1) - m * gamma * (v - 1) * ((gamma - 1) * (gamma * m + 1) * (v - 1) + (gamma + 1) * v) * alpha

    return np.array([v * F1, theta * F2, m * F3], dtype=np.float64)



def f_pendulum_double_system(X, p):
    return np.append(f_pendulum_raw_0_it(X, p, 0), f_pendulum_raw_0_it(X, p, 4))

def createFunc(p, kind=None):
    if not kind:
        return lambda t, y: f_regular(y, p)
    if kind == 'polynomial':
        p = polyfromroots(p)
        return lambda t, y: f_polynomial(y, p)
    if kind == 'target':
        return lambda t, y: f_target(y, p)
    if kind == 'pendulums':
        return lambda t, y: f_pendulum_double_system(y, p)
    if kind == '2ph_fluid':
        return lambda t, y: f_two_phase_fluid_in_pipe(y, p)

def getSolution(rk, iters):
    line = [rk.y]

    for i in range(1, iters):
        res = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = rk.step()
        if res != None or \
            np.isnan(rk.y).any() or \
            rk.status != "running":
            break

        line.append(rk.y)

        if np.isinf(rk.y).any():
            break

    if len(line) < iters:
        line += [line[-1]] * (iters - len(line))

    return line

def splitByNanInTwo(array):
    nan_index = np.where(np.isnan(array))
    nan_index = nan_index[0][0]
    return array[:nan_index], array[nan_index+1:]

# is a primitive solution's embedding extractor
def polarAngleExtractor(solutions):
    last_points = solutions[:, -1]
    complex_array = last_points[:, 0] + 1j * last_points[:, 1]
    return np.angle(complex_array)

# takes an array of 2xN arrays (paths)
# bg = (points, colors)
def drawSolutions(solutions, drawing_params=None, data_container=None):
    fig: Figure = plt.figure()

    plt.xlim(drawing_params['xmin'], drawing_params['xmax'])
    plt.ylim(drawing_params['ymin'], drawing_params['ymax'])

    x = [_[0] for _ in data_container['points']]
    y = [_[1] for _ in data_container['points']]
    maximal_confidence = data_container['m_conf']
    colors = data_container['colors']
    target_colors, target_labels = data_container.get('target_colors', ([], []))

    x_index = drawing_params['x_index']
    y_index = drawing_params['y_index']

    number_of_true_solutions = drawing_params.get('draw_last', 100)
    limit_true_solutions = True
    if number_of_true_solutions is None:
        number_of_true_solutions = len(solutions)
        limit_true_solutions = False

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    
    plt.scatter(x, y, c=colors, marker='x')

    for i in range(len(solutions) - number_of_true_solutions, len(solutions), 1):
        if i < 0:
            continue

        plt.scatter(solutions[i][0][x_index], solutions[i][0][y_index], marker='x')
        if limit_true_solutions:
            px, nx = splitByNanInTwo(solutions[i][:, x_index])
            py, ny = splitByNanInTwo(solutions[i][:, y_index])
            plt.plot(px, py, linestyle="-.", linewidth=1)
            plt.plot(nx, ny, linestyle=":", linewidth=1)

    convex_hull = data_container.get('convex_hull', None)
    if convex_hull is not None:
        plt.plot(*convex_hull, 'r--', linewidth=2)
    
    patches = []
    for single_color, single_label in zip(target_colors, target_labels):
        patches.append(mpatches.Patch(color=single_color, label=single_label))
    if len(patches) > 0:
        plt.legend(handles=patches, bbox_to_anchor=(0, -0.15))

    target_file = data_container.get('target_file', None)
    if target_file is None:
        plt.show()
    else:
        plt.savefig(fname=target_file, dpi=1000, bbox_inches='tight')
    plt.close('all')

def GT_base1_classifier(solutions, parameters):
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

def GT_base2_classifier(solutions, parameters):
    classes = []
    parameters = list(parameters)
    parameters.sort()
    interval_points = [float('-inf')] + parameters + [float('inf')]
    for solution in solutions:

        solution_classes_embbedding = [False] * len(interval_points)
        x0 = solution[0][0]

        is_bigger = lambda v, target: v > target
        value_index = next(i for i, v in enumerate(interval_points) if is_bigger(v, x0))
        solution_classes_embbedding[value_index] = True

        classes.append(solution_classes_embbedding)

    return 'regular', classes

def target_asymptotic_classifier(solutions, parameters):
    classes = []
    for solution in solutions:
        nan_index = np.where(np.isnan(solution))
        nan_index = nan_index[0][0]
        first_half = solution[:nan_index]
        second_half = solution[nan_index+1:]

        first_target_value = first_half[-1][1]
        second_target_value = second_half[-1][1]
        first_value_is_positive = first_target_value > 0
        second_value_is_positive = second_target_value > 0

        ground_true_vector = [False] * 4
        index = int(first_value_is_positive) + 2 * int(second_value_is_positive)
        ground_true_vector[index] = True

        classes.append(ground_true_vector)

    return 'regular', classes

def pendulums_asymptotic_classifier(solutions, parameters):
    classes = []
    for solution in solutions:
        #solution_divergence = solution[-1][:4] - solution[-1][4:]
        is_stable = np.max(np.abs(solution[-1][:2])) <= pi
        classes.append([is_stable, not is_stable])
    return 'regular', classes

def two_phase_fluid_classifier(solutions, parameters):
    classes = []

    gamma = parameters[0]
    kappa = parameters[1]
    s = parameters[2]
    alpha = parameters[3]

    vstar_1 = (gamma ** 2) * ((kappa + 1) * alpha - s * 0.5) - alpha * gamma * kappa * 0.5 - alpha + s * 0.5
    vstar_2 = (gamma ** 2) * ((kappa + 1) * alpha - s * 0.5) - alpha * 0.5 + gamma * (alpha * 0.5 - kappa - s * 0.5)
    vstar = vstar_1 / vstar_2
    cap_gamma = (gamma - 1) / gamma

    the_omgsobig_value = 5

    closeness_epsilon = 1e-3
    def relative_epsilon__(x, base_epsilon):
        return math.sqrt(1 + x ** 2) * base_epsilon
    def relative_epsilon(x):
        return relative_epsilon__(x, closeness_epsilon)

    def are_close(a, b):
        difference = abs(a - b)
        a_epsilon = relative_epsilon(a)
        b_epsilon = relative_epsilon(b)
        return difference < min(a_epsilon, b_epsilon)

    for solution in solutions:
        nan_index = np.where(np.isnan(solution))
        nan_index = nan_index[0][0]
        positive_half = solution[:nan_index]
        negative_half = solution[nan_index+1:]

        positive_v = positive_half[-1][0]
        negative_v = negative_half[-1][0]

        positive_theta = positive_half[-1][1]
        negative_theta = negative_half[-1][1]

        positive_m = positive_half[-1][2]
        negative_m = negative_half[-1][2]

        # positive part
        theta_expr1 = positive_v ** 2 / ( (positive_v ** 2) - alpha * gamma * (positive_v - 1) * (positive_v - cap_gamma) )

        p_1 = (positive_v ** 2) + (positive_m ** 2) + (positive_theta ** 2) < (closeness_epsilon ** 2)
        p_2 = are_close(1, positive_v) and are_close(1, positive_theta) and positive_m > 1
        p_3 = are_close(1, positive_m) and \
              are_close(theta_expr1, positive_theta) and \
              (1 < positive_v and positive_v < vstar)
        p_4 = positive_v > the_omgsobig_value or \
            positive_theta > the_omgsobig_value or \
            positive_m > the_omgsobig_value

        # negative part
        m_1_expr = ((s / kappa) + 1) / (kappa + 1) * ((s / kappa) + gamma)

        n_1 = are_close(negative_v, 1) and \
            are_close(negative_theta, 1) and \
            negative_m < m_1_expr
        n_2 = negative_v > the_omgsobig_value or \
            negative_theta > the_omgsobig_value or \
            negative_m > the_omgsobig_value

        ground_true_vector = [
            False ,
            p_1 and n_1 ,
            p_2 and n_1 ,
            p_3 and n_1 ,
            p_4 and n_1 ,
            p_1 and n_2 ,
            p_2 and n_2 ,
            p_3 and n_2 ,
            p_4 and n_2
            ]
        
        #if np.max(ground_true_vector) == 0:
        #    print([positive_v, negative_v, positive_m, negative_m, positive_theta, negative_theta])

        classes.append(ground_true_vector)

    return 'regular', classes

def makeSomeSolutions(rk_method,
                      function,
                      T_bound=None,
                      max_step=None,
                      rk_iterations=None,

                      paralelism=None,
                      proposed_points=None,
                      strip_except_first_and_last=None,

                      N=None,
                      dims=None,
                      vals_range=None,
                      cube_transformer=None):
    solutions = []

    if N is not None:
        proposed_points = [[random.uniform(vals_range[i][0], vals_range[i][1]) for i in range(dims)] for _ in range(N)]
        if proposed_points is not None:
            proposed_points = cube_transformer(proposed_points)

    def stripSolution(solution):
        if strip_except_first_and_last is None: 
            return solution
        begining = solution[:strip_except_first_and_last]
        ending = solution[-strip_except_first_and_last:]
        return begining + ending

    def getSingleSolutionWrap(x_i, rk_method, function, T_bound,
                              max_step, rk_iterations):
        rk = rk_method(function, 0, np.array(x_i, dtype=np.float64),
                       T_bound, max_step=max_step)

        new_solution = getSolution(rk, rk_iterations)
        return stripSolution(new_solution)

    def getBiderectionalSolutionWrap(x_i, rk_method, function, T_bound,
                                     max_step, rk_iterations):
        if not isinstance(T_bound, list):
            return getSingleSolutionWrap\
                (x_i, rk_method, function, T_bound, max_step, rk_iterations)

        solutions = []
        for bound_index, this_bound in enumerate(T_bound):
            solutions += getSingleSolutionWrap(
                x_i, rk_method, function, this_bound, max_step, rk_iterations)
            if bound_index != len(T_bound) - 1:
                solutions += [solutions[0] * nan]
        return np.array(solutions)

    begin = time.time()

    if paralelism is None:
        for x_i in tqdm(proposed_points):
            solutions.append(getBiderectionalSolutionWrap(
                x_i, rk_method, function, T_bound, max_step, rk_iterations))
    else:
        generator = Parallel(n_jobs=paralelism)(
            delayed(getBiderectionalSolutionWrap)
                (x_i, rk_method, function, T_bound, max_step, rk_iterations)
                    for x_i in proposed_points)
        solutions = [_ for _ in tqdm(generator)]

    total = time.time() - begin
    print(f"Elapsed time for solutions: {total} seconds")

    return solutions

def runClassificationTrain(solutions_embeddings, classes, pretrained_classifier=None, classifier_name=None, params=None):
    if not pretrained_classifier:
        if classifier_name == "knn":
            pretrained_classifier = \
                KNeighborsClassifier(n_neighbors=params["k"],
                                     n_jobs=params["cores_count"])
    pretrained_classifier.fit(solutions_embeddings, classes)
    return pretrained_classifier

def makeLinearCombinations(points, repeats):
    points = [np.array(_) for _ in points]
    new_points = []
    for _ in range(repeats):
        i = int(random.uniform(0, len(points)))
        j = int(random.uniform(0, len(points)))
        t = random.uniform(0, 1)
        new_points.append((np.array(points[i])*t + (1. - t)*np.array(points[j])))
    return points + new_points

def boundingBoxNumpy(points):
    return [np.min(points, axis=0), np.max(points, axis=0)]

def makeCubePointsFromRanges(ranges, internal_steps=10, cube_transformer=None):
    ranges = np.array(ranges)

    size = internal_steps
    t_values = [(float(i))/size for i in range(size + 1)]
    #t_values = [x ** 0.75 for x in t_values]

    ranges_combination = [ranges[:,0] * t + (1. - t) * ranges[:,1] for t in t_values]
    unique_ranges_combination = np.unique(ranges_combination, axis=0)
    #unique_ranges_combination = [_ for _ in unique_ranges_combination]

    changing_indexes = []
    aggregate_template_point = []
    for i in range(len(ranges)):
        column_doesnt_change = np.allclose(unique_ranges_combination[:,i], unique_ranges_combination[:,i][0])
        if not column_doesnt_change:
            changing_indexes.append(i)
            aggregate_template_point.append(0.)
        else:
            aggregate_template_point.append(unique_ranges_combination[0][i])
    compressed_ranges_combination = unique_ranges_combination[:, changing_indexes]
    points = np.array(list(itertools.product(*zip(*compressed_ranges_combination))))

    unique_points = []
    for p in points:
        new_point = np.array(aggregate_template_point)
        new_point[changing_indexes] = p
        unique_points.append(new_point)

    if cube_transformer is not None:
        unique_points = cube_transformer(unique_points)

    return unique_points

def makeFinerMesh(points, total_range_steps=10, cube_transformer=None):
    bbox = boundingBoxNumpy(points)
    bbox[0][4:]=0
    bbox[1][4:]=0
    dimesnionality = len(bbox[0])
    ranges = [[bbox[0][index], bbox[1][index]] for index in range(dimesnionality)]
    return makeCubePointsFromRanges(ranges, total_range_steps, cube_transformer)

def normalize(points, ranges):
    def conditionalTransform(p, i, ranges):
        if ranges[i][0] == ranges[i][1]:
            return 0
        return (p[i] - ranges[i][0])/(ranges[i][1] - ranges[i][0])

    singlepoint_normalise = lambda p: [conditionalTransform(p, i, ranges) for i in range(len(ranges))]
    return np.array([singlepoint_normalise(pt) for pt in points])

def denormalize(points, ranges):
    def conditionalTransform(p, i, ranges):
        if ranges[i][0] == ranges[i][1]:
            return ranges[i][0]
        return p[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]

    singlepoint_normalise = lambda p: [conditionalTransform(p, i, ranges) for i in range(len(ranges))]
    return np.array([singlepoint_normalise(pt) for pt in points])

def classesProbabilitiesToSingularValues(class_probabilities):
    number_of_classes = len(class_probabilities[0])

    hue_values = np.linspace(0, 1, number_of_classes + 1)
    value_shift = 0.5 / (number_of_classes)
    colors = []
    max_probs = []

    for single_element in class_probabilities:
        normalised_probabilites = single_element / np.sum(single_element)
        current_color = np.array([0,0,0], dtype=np.float64)
        for i in range(len(normalised_probabilites)):
            current_color += \
                np.array(colorsys.hsv_to_rgb(
                    hue_values[i] + value_shift,
                    1 - 0.75 * (normalised_probabilites[i] ** 2),
                    normalised_probabilites[i] ** .5)) * normalised_probabilites[i]
        
        max_prob = np.max(normalised_probabilites)

        colors.append(current_color)
        max_probs.append(max_prob)
        #max_probs.append(normalised_probabilites[0])
    return np.array(colors), np.array(max_probs)

class CMeansWrapper:
    def __init__(self, points_count, params={}):
        None

    def fit_predict(self, points):
        None

class SOODE_AC_Core:
    def __init__(self, SOODE_kind=None, SOODE_params={}):
        self.SOODE_kind = SOODE_kind
        solution_classifier = \
            lambda _1, _2, _3: (f"{SOODE_kind}@unimplimented", [[False]] * len(_1))
        self.solution_embedding_extractor = \
            lambda x: np.array(x)[:,0]
        SOODE_solver = RK23

        temp = SOODE_params.get('SOODE_parameters', None)
        if not temp:
            random_params_count = SOODE_params.get('SOODE_random_parameters_count', 5)
            temp = [random.uniform(-10, 10) for _ in range(random_params_count)]
        self.SOODE_parameters = temp
        print(self.SOODE_parameters)

        temp = SOODE_params.get('initial_solutions_count', None)
        if not temp:
            temp = 150
        self.initial_solutions_count = int(temp)

        initial_region_ranges = SOODE_params.get('initial_region_ranges', None)

        temp = SOODE_params.get('classifier_type', None)
        if not temp:
            temp = 'knn'
        self.ml_classifier_type = temp

        self.ml_classifier_params = SOODE_params.get('classifier_params', None)
        self.ml_linear_combinations_density = int(SOODE_params.get('linear_combinations_density', 10000))
        self.ml_accuracy_threshold = float(SOODE_params.get('accuracy_threshold', 1.))
        drawing_params = SOODE_params.get('drawing_params', {})
        T_bound = SOODE_params.get('t_bound', 100)

        initial_cube_transformer = None
        if not SOODE_kind:
            self.SOODE_dims = 2
            self.params_count = 4
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            drawing_params['classes_labels'] = ["1", "2", "3", "4"]
            self.solution_classifier = lambda _1, _2, _3: GT_base1_classifier(_1, _2)
            T_bound = [T_bound, -T_bound]
        elif SOODE_kind == 'polynomial':
            self.SOODE_dims = 2
            self.params_count = len(self.SOODE_parameters)
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            self.solution_classifier = lambda _1, _2, _3: GT_base2_classifier(_1, _2)
        elif SOODE_kind == 'target':
            self.SOODE_dims = 3
            self.params_count = 1
            SOODE_solver = RK45
            drawing_params['classes_labels'] = ["--", "+-", "-+", "++"]
            self.solution_classifier = lambda _1, _2, _3: target_asymptotic_classifier(_1, _2)
            T_bound = [T_bound, -T_bound]
        elif SOODE_kind == 'pendulums':
            self.SOODE_dims = 8
            self.params_count = 4
            SOODE_solver = DOP853
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            drawing_params['classes_labels'] = ["stable", "unstable"]
            def init_cube_transf_func(c):
                c = np.array(c)
                eps = self.SOODE_parameters[3]
                if c.shape[1] == 4:
                    return np.append(c, c + [eps, eps, 0, 0], axis=1)
                c[:,4] = c[:,0] + eps
                c[:,5] = c[:,1] + eps
                c[:,6] = c[:,2]
                c[:,7] = c[:,3]
                return c

            initial_cube_transformer = init_cube_transf_func
            self.solution_classifier = lambda _1, _2, _3: pendulums_asymptotic_classifier(_1, _2)
        elif SOODE_kind == '2ph_fluid':
            self.SOODE_dims = 3
            self.params_count = 4
            SOODE_solver = RK45
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            drawing_params['classes_labels'] = \
                ["none of the bellow",
                    "+ m,t,v~0;; - t,v~1 m<M",
                    "+ t,v~1 m>1;; - t,v~1 m<M",
                    "+ m~1 t~T 1<v<V*;; - t,v~1 m<M",
                    "+ t,v~1 m>1;; - t,v~1 m<M",
                    "+ m,t,v~0;; - m or t or v->inf",
                    "+ t,v~1 m>1;; - m or t or v->inf",
                    "+ m~1 t~T 1<v<V*;; - m or t or v->inf",
                    "+ t,v~1 m>1;; - m or t or v->inf"]
            self.solution_classifier = lambda _1, _2, _3: two_phase_fluid_classifier(_1, _2)
            T_bound = [T_bound, -T_bound]
        else:
            self.solution_classifier = solution_classifier
            None
            ### ... ?

        if not initial_region_ranges:
            initial_region_ranges = [[-20, 20]] * self.SOODE_dims

        fine_mesh_steps = SOODE_params.get('fine_mesh_steps', 160)
        if fine_mesh_steps is None:
            fine_mesh_steps = int(self.ml_linear_combinations_density ** (1. / self.SOODE_dims) + 1)
        self.fine_mesh_steps = fine_mesh_steps

        drawing_ranges = [
            [drawing_params['xmin'], drawing_params['xmax']],
            [drawing_params['ymin'], drawing_params['ymax']]
        ]

        self.cube_transformer = None
        self.initial_cube_transformer = initial_cube_transformer or (lambda x: x)
        self.base_initial_cube_points = makeCubePointsFromRanges(initial_region_ranges, cube_transformer=initial_cube_transformer)
        self.base_drawing_plane_points = np.array(
            makeCubePointsFromRanges(drawing_ranges,
                                     drawing_params.get('resolution', 160),
                                     cube_transformer=self.cube_transformer))
        self.initial_region_ranges = initial_region_ranges
        self.drawing_params = drawing_params
        self.drawing_ranges = drawing_ranges
        self.drawing_counter = 0

        self.solver_iterations = SOODE_params.get('solver_iterations', 100)
        self.T_bound = T_bound
        self.solver_max_step = inf
        self.SOODE_solver = SOODE_solver
        self.SOODE_func = createFunc(self.SOODE_parameters, SOODE_kind)

        self.true_solutions = None
        self.true_solutions_classes = None

        self.enable_debug_plots = SOODE_params.get('enable_debug_plots', False)

        self.classifier = None
        self.strip_solutions_except = SOODE_params.get('strip_down_to', None)
        self.classifer_params = self.ml_classifier_params

        clustrizer = None
        if SOODE_params.get('__use_c_means', False):
            clustrizer = CMeansWrapper(SOODE_params['__clustering_n'])
        else:
            clustrizer = MiniBatchKMeans(SOODE_params['__clustering_n'])
        self.clustrizer = clustrizer
        self.rk_paralelism = SOODE_params['__rk_paralelism']

        self.points_for_drawing = np.array([[]])
        self.points_classes = np.array([[]])

        self.prev_iteration_proposed_points = None
        self.preprev_iter_proposed_points = self.base_initial_cube_points

        self.drawing_iter_data = None

    def addSaltsToSolutions(self, new_solutions, classes):
        #mx = np.max(new_solutions, axis=0)
        mn = np.min(new_solutions, axis=0)
        point = mn
        classes_count = len(classes[0])
        for i in range(classes_count):
            classes.append([i == j for j in range(classes_count)])
            new_solutions.append(mn)
        
    def runOneIteration(self):
        if self.prev_iteration_proposed_points is None:
            new_solutions = makeSomeSolutions(self.SOODE_solver,
                                              self.SOODE_func,
                                              T_bound=self.T_bound,
                                              max_step=self.solver_max_step,
                                              rk_iterations=self.solver_iterations,
                                              paralelism=self.rk_paralelism,
                                              strip_except_first_and_last=self.strip_solutions_except,
                                              N=self.initial_solutions_count,
                                              dims=self.SOODE_dims,
                                              vals_range=self.initial_region_ranges,
                                              cube_transformer=self.initial_cube_transformer)
        else:
            new_solutions = makeSomeSolutions(self.SOODE_solver,
                                              self.SOODE_func,
                                              proposed_points=self.prev_iteration_proposed_points,
                                              paralelism=self.rk_paralelism,
                                              strip_except_first_and_last=self.strip_solutions_except,
                                              T_bound=self.T_bound,
                                              max_step=self.solver_max_step,
                                              rk_iterations=self.solver_iterations)

        detected_type, classes = self.solution_classifier(new_solutions, self.SOODE_parameters, self.SOODE_kind)

        if len(classes) == 0:
            return

        if self.prev_iteration_proposed_points is None:
            print(detected_type)
            self.addSaltsToSolutions(new_solutions, classes)

        numpyfied_classes = np.argmax(classes, axis=1)
        print(np.unique(numpyfied_classes, return_counts=True, axis=0))

        if self.true_solutions is not None:
            numpyfied_new_solutions = np.asarray(new_solutions)
            self.true_solutions = np.append(self.true_solutions, numpyfied_new_solutions, axis=0)
            self.true_solutions_classes = np.append(self.true_solutions_classes, numpyfied_classes, axis=0)
        else:
            self.true_solutions = np.array(new_solutions)
            self.true_solutions_classes = numpyfied_classes

        solutions_embeddings = normalize(self.solution_embedding_extractor(self.true_solutions), self.initial_region_ranges)

        self.classifier = runClassificationTrain(solutions_embeddings,
                                                 self.true_solutions_classes,
                                                 pretrained_classifier=self.classifier,
                                                  classifier_name=self.ml_classifier_type,
                                                 params=self.ml_classifier_params)

        if self.prev_iteration_proposed_points is None:
            whole_region_points = self.base_initial_cube_points
            new_ml_state_points_by_regions = [whole_region_points]
        else:

            predicted_clusters = self.clustrizer.fit_predict(self.prev_iteration_proposed_points)

            plt.scatter(self.prev_iteration_proposed_points[:,0], self.prev_iteration_proposed_points[:,1], c=predicted_clusters)

            new_ml_state_points_by_regions = [[] for _ in range(self.clustrizer.n_clusters)]
            for array_index, cluster_index in enumerate(predicted_clusters):
                new_ml_state_points_by_regions[cluster_index].append(
                    self.prev_iteration_proposed_points[array_index])

        pool_of_lcopps = None

        if self.enable_debug_plots:
            fig: Figure = plt.figure()

        colors = []
        for index, linear_combinations_of_proposed_points in enumerate(new_ml_state_points_by_regions):
            mesh_points_in_the_region = makeFinerMesh(linear_combinations_of_proposed_points,
                                                      self.fine_mesh_steps,
                                                      self.initial_cube_transformer)
            mesh_points_in_the_region = makeLinearCombinations(mesh_points_in_the_region, self.ml_linear_combinations_density)
            linear_combinations_of_proposed_points = np.append(linear_combinations_of_proposed_points, mesh_points_in_the_region, axis=0)

            normalised_lcopp = normalize(linear_combinations_of_proposed_points, self.initial_region_ranges)
            probs = self.classifier.predict_proba(normalised_lcopp)

            filtered_lcopps = linear_combinations_of_proposed_points[(probs < self.ml_accuracy_threshold).all(axis=1)]

            if pool_of_lcopps is not None:
                pool_of_lcopps = np.append(filtered_lcopps, pool_of_lcopps, axis=0)
            else:
                pool_of_lcopps = filtered_lcopps

            colors += [index] * filtered_lcopps.shape[0]

        self.prev_iteration_proposed_points = pool_of_lcopps
        print(f'Proposed points count: {len(self.prev_iteration_proposed_points)}')

        if self.enable_debug_plots:
            plt.scatter(pool_of_lcopps[:,0],
                        pool_of_lcopps[:,1],
                        c=colors)

            plt.savefig(fname=f'lcopps/lcopps{self.drawing_counter}.png')
            plt.close('all')
            self.drawing_counter += 1

    def buildLabelSet(self, labels, classes_count):
        labels = deepcopy(labels)[:classes_count]
        fetch_array = []
        
        for i in range(classes_count):
            new_zero_array = [0] * classes_count
            new_zero_array[i] = 1
            fetch_array.append(new_zero_array)

        fetch_array.append([0] * classes_count)
        labels.append("undefined")

        colors, _ = classesProbabilitiesToSingularValues(fetch_array)
        return colors, labels

    def drawCurrentState(self, plot_filename=None):
        if len(self.prev_iteration_proposed_points) == 0:
            raise Exception("That's all folks!")

        x_index = self.drawing_params['x_index']
        y_index = self.drawing_params['y_index']
        labels = self.drawing_params.get('classes_labels', [])

        if self.SOODE_dims > 2:
            slice_coordinates = self.drawing_params.get('slice_coords', None)
            if not slice_coordinates:
                slice_center = np.average(self.prev_iteration_proposed_points, axis=0)
                slice_coordinates = [x for i, x in enumerate(slice_center) if i not in (x_index, y_index)]

            slice_coordinates_repeated = np.repeat([slice_coordinates], len(self.base_drawing_plane_points), axis=0)
            if x_index <= y_index:
                # np.insert(np.insert(np.repeat([q], 2, axis=0), 1, a[:,0], axis=1), 3, a[:,1], axis=1)
                x_slice = np.expand_dims(self.base_drawing_plane_points[:,0], axis=1)
                y_slice = np.expand_dims(self.base_drawing_plane_points[:,1], axis=1)
                flattened_points_for_preview = np.insert(slice_coordinates_repeated, [x_index], x_slice, axis=1)
                flattened_points_for_preview = np.insert(flattened_points_for_preview, [y_index], y_slice, axis=1)
            else:
                x_slice = np.expand_dims(self.base_drawing_plane_points[:,0], axis=1)
                y_slice = np.expand_dims(self.base_drawing_plane_points[:,1], axis=1)
                flattened_points_for_preview = np.insert(slice_coordinates_repeated, [y_index], y_slice, axis=1)
                flattened_points_for_preview = np.insert(flattened_points_for_preview, [x_index], x_slice, axis=1)
        else:
            flattened_points_for_preview = self.base_drawing_plane_points

        flattened_normalised_points_for_preview = normalize(flattened_points_for_preview, self.initial_region_ranges)
        probs = self.classifier.predict_proba(flattened_normalised_points_for_preview)
        colors, max_confidence  = classesProbabilitiesToSingularValues(probs)
        label_colors, labels = self.buildLabelSet(labels, probs.shape[1])

        total_incorrect = (max_confidence < self.ml_accuracy_threshold).sum()
        total_size = np.prod(max_confidence.shape)
        p_area = float(total_size - total_incorrect) / total_size

        print(f"PA: {p_area}, total solutions {self.true_solutions.shape[0]}")

        proposed_points_proj = self.prev_iteration_proposed_points[:, [x_index, y_index]]
        #hull = ConvexHull(proposed_points_proj)

        drawSolutions(
              self.true_solutions,
              self.drawing_params,
              {
                  'points': self.base_drawing_plane_points,
                  'm_conf': max_confidence,
                  'colors': colors,
                  'target_colors': (label_colors, labels),
                  #'convex_hull': (proposed_points_proj[hull.vertices,0], proposed_points_proj[hull.vertices,1]),
                  'target_file': plot_filename
              })

SOODE_AC_instance = SOODE_AC_Core(
    SOODE_kind='2ph_fluid',
    SOODE_params={
        'classifier_params': {
            'k': 20,
            'cores_count': 12
        },
        'enable_debug_plots': False,
        'initial_region_ranges': [[0.5, 1.5], [0.5, 1.5], [0.5, 1.5]],
        'SOODE_parameters': [1.5, 0.2, 0.25, 0.21],
        'accuracy_threshold': 1 - 1e-4,
        'initial_solutions_count': 300,
        'fine_mesh_steps': 30,
        'linear_combinations_density': 1500,
        'classifier_type': 'knn',
        'solver_iterations': 3333,
        'strip_down_to': 5,
        't_bound': 1e10,
        '__clustering_n': 10,
        '__rk_paralelism': 10,
        'drawing_params': {
            'slice_coords': [0.99],
            'xmin': 0.5,
            'xmax': 1.5,
            'ymin': 0.5,
            'ymax': 1.5,
            'x_index': 0,
            'y_index': 1,
            'draw_last': None,
            'resolution': 500
        }
    })
## 
for i in range(100):
    SOODE_AC_instance.runOneIteration()
    SOODE_AC_instance.drawCurrentState(f"{i}_fig.png")

    joblib.dump(SOODE_AC_instance.classifier, f'_nn_iter{i}.pkl')
    #modelscorev2 = joblib.load('scoreregression.pkl' , mmap_mode ='r')
