import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from typing import List
from ctypes import *

linearDll = CDLL("./MachineLearning_Semester1.dll")

points = np.array([
    [1, 1],
    [2, 1],
    [2, 2]
])
classes = np.array([
    1,
    1,
    -1
])

colors = ['blue' if c == 1 else 'red' for c in classes]

w = np.random.uniform(-1.0, 1.0, 3)
print(w)

doublePTR = POINTER(c_double)
intPTR = POINTER(c_int)
floatPTR = POINTER(c_float)

points_ptr = doublePTR.from_buffer(points)
classes_ptr = doublePTR.from_buffer(classes)
w_ptr = doublePTR.from_buffer(w)

# LINEAR TRAINING

linearDll.linearTraining.argtypes = [POINTER(c_double), POINTER(c_double), c_int, c_int, POINTER(c_double), c_double]
linearDll.linearTraining.restype = POINTER(c_double)

linearDll.linearTraining(points_ptr, classes_ptr, len(points), 10000, w_ptr, 3)

print(w)

test_points = []
test_colors = []
for row in range(0, 300):
    for col in range(0, 300):
        p = np.array([col / 100, row / 100])
        c = 'lightcyan' if np.matmul(np.transpose(w), np.array([1.0, *p])) >= 0 else 'pink'
        test_points.append(p)
        test_colors.append(c)
test_points = np.array(test_points)
test_colors = np.array(test_colors)

plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.show()

# LINEAR TRAINING


# PMC TRAINING

neuronsTab_array = np.array([2, 1])
sizeNeuronTab = neuronsTab_array.size
maxNumberLayer = 2
deltas_array = np.copy(points)
deltas = floatPTR.from_buffer(deltas_array)
neuronsTab = intPTR.from_buffer(neuronsTab_array)

linearDll.initPMC.argtypes = [POINTER(c_int), c_int, c_int, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
linearDll.initPMC(neuronsTab, sizeNeuronTab, maxNumberLayer, points_ptr, deltas, w_ptr)

linearDll.PMCTraining.argtypes = [points_ptr, points.size, classes_ptr, classes.size, True, [[c] for c in classes],
                                  classes.size, maxNumberLayer, points_ptr, deltas, w_ptr]

test_points = []
test_colors = []
for row in range(0, 300):
  for col in range(0, 300):
    p = np.array([col / 100, row / 100])
    c = 'lightcyan' if linearDll.predictPMC(p, True)[0] >= 0 else 'pink'
    test_points.append(p)
    test_colors.append(c)
test_points = np.array(test_points)
test_colors = np.array(test_colors)

plt.scatter(test_points[:, 0], test_points[:, 1], c=test_colors)
plt.scatter(points[:, 0], points[:, 1], c=colors)
plt.show()