import pandas as pd
import numpy as np

a = np.array([[1,1,1],
     [2,1,0],
     [0,0,0]])
print(np.sum(a>0,axis=0))