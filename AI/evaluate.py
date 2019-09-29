import numpy as np
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


f = h5py.File('./food101/food_c101_n1000_r384x384x3.h5', 'r')

x = np.array(f["images"])/255.
y = np.array([[int(i) for i in f["category"][j]] for j in range(len(f["category"]))])

train_x,test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)