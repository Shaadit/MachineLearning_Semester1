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
    0,
    -1
])

colors = ['blue' if c == 1 else 'red' for c in classes]

w = np.random.uniform(-1.0, 1.0, 3)
print(w)

doublePTRPTR = POINTER(POINTER(c_double))
points_ptr = doublePTRPTR.from_buffer(points)

doublePTR = POINTER(c_double)
classes_ptr = doublePTR.from_buffer(classes)
w_ptr = doublePTR.from_buffer(w)

print(points_ptr.__sizeof__())



linearDll.linearTraining.argtypes = [POINTER(POINTER(c_double)), POINTER(c_double), c_int, c_int, POINTER(c_double)]
linearDll.linearTraining.restype = POINTER(c_float)

x = linearDll.linearTraining(points_ptr, classes_ptr, len(points), 10000, w_ptr)

# x2 = (np.ctypeslib.as_array(x, shape=(3, 1)))

print(x)

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
print("success")