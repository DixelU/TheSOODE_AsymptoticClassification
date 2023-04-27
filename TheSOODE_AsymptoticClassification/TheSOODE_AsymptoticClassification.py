from cmath import inf
import itertools
import numpy as np
import math
import pylab as pl
import warnings
import random
import re
import colorsys
import matplotlib
import seaborn as sns

from tqdm import tqdm
from matplotlib.figure import Figure
from scipy.integrate import RK45, RK23, DOP853
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.spatial import ConvexHull
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyval, polyfromroots

#from numba import njit
matplotlib.use('Agg')

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

def createFunc(p, kind=None):
    if not kind:
        return lambda t, y: f_regular(y, p) 
    if kind == 'polynomial':
        p = polyfromroots(p)
        return lambda t, y: f_polynomial(y, p)
    if kind == 'target':
        return lambda t, y: f_target(y, p)

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
def drawSolutions(solutions, drawing_params=None, data_container=None):
    fig: Figure = plt.figure()

    plt.xlim(drawing_params['xmin'], drawing_params['xmax'])
    plt.ylim(drawing_params['ymin'], drawing_params['ymax'])
    
    x = [_[0] for _ in data_container['points']]
    y = [_[1] for _ in data_container['points']]
    maximal_confidence = data_container['m_conf']
    
    x_index = drawing_params['x_index']
    y_index = drawing_params['y_index']

    numer_of_true_solutions = drawing_params.get('draw_last', 100)
    limit_true_solutions = True
    if numer_of_true_solutions is None: 
        numer_of_true_solutions = len(solutions)
        limit_true_solutions = False

    ax = plt.gca()
    #center = ax.tricontourf(x, y, maximal_confidence, levels=20, linewidths=0.5, colors="k")
    center = ax.tricontourf(x, y, maximal_confidence, levels=20, cmap="PuBu_r", antialiased=False)
    fig.colorbar(center, ax=ax)

    for i in range(len(solutions) - numer_of_true_solutions, len(solutions), 1):
        plt.scatter(solutions[i][0][x_index], solutions[i][0][y_index], marker='x')
        if limit_true_solutions:
            plt.plot(solutions[i][:, x_index], solutions[i][:, y_index], linestyle=":", linewidth=1)

    convex_hull = data_container.get('convex_hull', None)
    if convex_hull is not None:
        plt.plot(*convex_hull, 'r--', linewidth=2)

    target_file = data_container.get('target_file', None)
    if target_file is None:
        plt.show()
    else:
        plt.savefig(fname=target_file)
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
        solution_classes_embbedding = [False]
        classes.append(solution_classes_embbedding)

    return 'regular', classes

def makeSomeSolutions(rk_method,
                      function,
                      T_bound=None,
                      max_step=None,
                      rk_iterations=None,

                      paralelism=None,
                      proposed_points=None,

                      N=None,
                      dims=None,
                      vals_range=None):
    solutions = []

    if N:
        proposed_points = [[random.uniform(vals_range[i][0], vals_range[i][1]) for i in range(dims)] for _ in range(N)]

    def exec(x_i, rk_method, function, T_bound, max_step, rk_iterations):
        rk = rk_method(function, 0, np.array(x_i, dtype=np.float64), T_bound, max_step=max_step)

        new_solution = getSolution(rk, rk_iterations)
        if np.isnan(new_solution).any():
            return None
        return new_solution

    if paralelism is None:
        for x_i in tqdm(proposed_points):
            solutions.append(exec(x_i, rk_method, function, T_bound, max_step, rk_iterations))
    else: 
        generator = Parallel(n_jobs=paralelism)(
            delayed(exec)(x_i, rk_method, function, T_bound, max_step, rk_iterations) for x_i in proposed_points)
        solutions = [_ for _ in tqdm(generator)]

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

def makeCubePointsFromRanges(ranges, internal_steps=10):
    ranges = np.array(ranges)
    
    size = internal_steps
    t_values = [float(i)/size for i in range(size + 1)]
    
    ranges_combination = [ranges[:,0] * t + (1. - t) * ranges[:,1] for t in t_values]
    return list(itertools.product(*zip(*ranges_combination)))

def makeFinerMesh(points, linear_repeats):
    bbox = boundingBoxNumpy(points)
    dimesnionality = len(bbox[0])
    ranges = [[bbox[0][index], bbox[1][index]] for index in range(dimesnionality)]
    return makeCubePointsFromRanges(ranges, int(math.ceil(linear_repeats ** (1 / dimesnionality))))

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
        self.ml_accuracy_threshold = float(SOODE_params.get('linear_combinations_density', 1.))
        drawing_params = SOODE_params.get('drawing_params', {})

        if not SOODE_kind:
            self.SOODE_dims = 2
            self.params_count = 4
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            self.solution_classifier = lambda _1, _2, _3: GT_base1_classifier(_1, _2)
        elif SOODE_kind == 'polynomial':
            self.SOODE_dims = 2
            self.params_count = len(self.SOODE_parameters)
            drawing_params['x_index'] = 0
            drawing_params['y_index'] = 1
            self.solution_classifier = lambda _1, _2, _3: GT_base2_classifier(_1, _2)
        elif SOODE_kind == 'target':
            self.SOODE_dims = 3
            self.params_count = 1
            SOODE_solver = DOP853
            self.solution_classifier = lambda _1, _2, _3: target_asymptotic_classifier(_1, _2)
        else:
            self.solution_classifier = solution_classifier
            None
            ### ... ?

        if not initial_region_ranges:
            self.initial_region_ranges = [[-20,20]] * self.SOODE_dims
        else:
            self.initial_region_ranges = initial_region_ranges

        drawing_ranges = [
            [drawing_params['xmin'], drawing_params['xmax']],
            [drawing_params['ymin'], drawing_params['ymax']]    
        ]

        cube_points = makeCubePointsFromRanges(drawing_ranges)
        self.base_initial_cube_points = makeCubePointsFromRanges(self.initial_region_ranges)
        self.base_drawing_plane_points = np.array(makeFinerMesh(cube_points, 
                                                                drawing_params.get('resolution', 25000)))
        #self.base_drawing_normalised_plane_points = \
        #    normalize(self.base_drawing_plane_points, drawing_ranges)
        self.drawing_params = drawing_params
        self.drawing_ranges = drawing_ranges
            
        self.linear_combinations_after_selection = SOODE_params['linear_combinations_after_selection']
        self.solver_iterations = 25
        self.T_bound = 100
        self.solver_max_step = 1
        self.SOODE_solver = SOODE_solver
        self.SOODE_func = createFunc(self.SOODE_parameters, SOODE_kind)
        
        self.true_solutions = None
        self.true_solutions_classes = None

        self.classifier = None
        self.classifer_params = self.ml_classifier_params
        self.clustrizer = MiniBatchKMeans(
            SOODE_params['__clustering_n']
        )
        self.rk_paralelism = SOODE_params['__rk_paralelism']

        self.points_for_drawing = np.array([[]])
        self.points_classes = np.array([[]])

        self.prev_iteration_proposed_points = None
        self.preprev_iter_proposed_points = self.base_initial_cube_points

        self.drawing_iter_data = None

    def runOneIteration(self, ):
        if self.prev_iteration_proposed_points is None:
            new_solutions = makeSomeSolutions(self.SOODE_solver,
                                              self.SOODE_func,
                                              T_bound=self.T_bound,
                                              max_step=self.solver_max_step,
                                              rk_iterations=self.solver_iterations,
                                              N=self.initial_solutions_count,
                                              dims=self.SOODE_dims,
                                              vals_range=self.initial_region_ranges)
        else:
            new_solutions = makeSomeSolutions(self.SOODE_solver,
                                              self.SOODE_func,
                                              proposed_points=self.prev_iteration_proposed_points,
                                              paralelism=self.rk_paralelism,
                                              T_bound=self.T_bound,
                                              max_step=self.solver_max_step,
                                              rk_iterations=self.solver_iterations)
            
        detected_type, classes = self.solution_classifier(new_solutions, self.SOODE_parameters, self.SOODE_kind)
        
        if self.prev_iteration_proposed_points is None: 
            print(detected_type)
            
        numpyfied_classes = np.argmax(classes, axis=1)

        if self.true_solutions is not None:
            self.true_solutions = np.append(self.true_solutions, new_solutions, axis=0)
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
        
        #self.prev_iteration_proposed_points = None
        if self.prev_iteration_proposed_points is None:
            whole_region_points = self.base_initial_cube_points#makeFinerMesh(self.base_initial_cube_points, self.ml_linear_combinations_density)
            new_ml_state_points_by_regions = [whole_region_points]
        else:
            #new_near_proposed_points = makeFinerMesh(
            #    #np.append(self.prev_iteration_proposed_points, self.preprev_iter_proposed_points, axis=0),
            #    self.prev_iteration_proposed_points,
            #    self.ml_linear_combinations_density)

            predicted_clusters = self.clustrizer.fit_predict(self.prev_iteration_proposed_points)
            
            new_ml_state_points_by_regions = [[] for _ in range(self.clustrizer.n_clusters)]
            for array_index, cluster_index in enumerate(predicted_clusters):
                new_ml_state_points_by_regions[cluster_index].append(
                    self.prev_iteration_proposed_points[array_index])
            self.preprev_iter_proposed_points = self.prev_iteration_proposed_points

        local_points_pool = None
        local_points_pool_classes = None
        pool_of_lcopps = None
        
        fig: Figure = plt.figure()

        for linear_combinations_of_proposed_points in new_ml_state_points_by_regions:
            mesh_points_in_the_region = makeFinerMesh(linear_combinations_of_proposed_points, self.ml_linear_combinations_density)
            linear_combinations_of_proposed_points = np.append(linear_combinations_of_proposed_points, mesh_points_in_the_region, axis=0)

            normalised_lcopp = normalize(linear_combinations_of_proposed_points, self.initial_region_ranges)
            probs = self.classifier.predict_proba(normalised_lcopp)

            if local_points_pool is not None:
                local_points_pool = np.append(local_points_pool, normalised_lcopp, axis=0)
                local_points_pool_classes = np.append(local_points_pool_classes, probs, axis=0)
                linear_combinations_of_proposed_points = np.array(linear_combinations_of_proposed_points)
                pool_of_lcopps = np.append(linear_combinations_of_proposed_points, pool_of_lcopps, axis=0) 
            else: 
                local_points_pool = normalised_lcopp
                local_points_pool_classes = probs
                linear_combinations_of_proposed_points = np.array(linear_combinations_of_proposed_points)
                pool_of_lcopps = linear_combinations_of_proposed_points
            
            filtered_lcopps = linear_combinations_of_proposed_points[(probs < self.ml_accuracy_threshold).all(axis=1)]
            plt.scatter(filtered_lcopps[:,0], filtered_lcopps[:,1])

        self.prev_iteration_proposed_points = pool_of_lcopps[(local_points_pool_classes < self.ml_accuracy_threshold).all(axis=1)]
        self.prev_iteration_proposed_points = np.array(makeLinearCombinations(self.prev_iteration_proposed_points, 100))
        
        plt.savefig(fname='lcopps/lcopps.png')

    def drawCurrentState(self, save_to_file=None):
        x_index = self.drawing_params['x_index']
        y_index = self.drawing_params['y_index']

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
        
        proposed_points_proj = self.prev_iteration_proposed_points[:, [x_index, y_index]]
        hull = ConvexHull(proposed_points_proj)
        
        drawSolutions(
              self.true_solutions,
              self.drawing_params,
              {
                  'points': self.base_drawing_plane_points, 
                  'm_conf': max_confidence,
                  'colors': colors,
                  #'convex_hull': (proposed_points_proj[hull.vertices,0], proposed_points_proj[hull.vertices,1]),
                  'target_file': save_to_file
              })
        
SOODE_AC_instance = SOODE_AC_Core(
    SOODE_kind='polynomial',
    SOODE_params={
        'classifier_params': {
            'k': 20,
            'cores_count': 6
        },
        '__clustering_n': 15,
        '__rk_paralelism': 6,
        'linear_combinations_after_selection': 100,
        'drawing_params': {
            'xmin': -20,
            'xmax': 20,
            'ymin': -20,
            'ymax': 20,
            'x_index': 0,
            'y_index': 1,
            'epsilon': 5,
            'draw_last': 150
        }
    })

for i in range(100):
    SOODE_AC_instance.runOneIteration()
    SOODE_AC_instance.drawCurrentState(f"{i}_fig.png")