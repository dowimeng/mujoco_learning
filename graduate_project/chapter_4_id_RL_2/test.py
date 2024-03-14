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

a = np.array([1,2,3])
b = np.ones(3) * 2
print(np.max(np.array((a,b)),axis=0))
c = np.array((a,b))
print(np.linalg.norm(c, axis=1))