import matplotlib.pyplot as plt
import numpy as np
import time

# a = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]
# b = np.ones((3,3))
# # print(a)
# # print(b)
# print(a-b)
joints_pos_limit = np.array([[-0.3, 0.3],
                             [-0.3, 0.3],
                             [-0.3, 0.3],
                             [-1.9, -0.9],
                             [-0.1, 0.1],
                             [1.4, 1.7],
                             [-1.3, -0.3], ])

a = np.array([1,2,3])
b = np.array([2])
c = np.array([5,6])
# print(a.tolist())
# print(b.tolist())
c = np.append(a,b)
print(c)

