from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing

data = np.loadtxt("yeast.data", usecols = (1,2,3,4,5,6,7,8))
label = np.loadtxt("yeast.data", usecols = (9),dtype=str)
le = preprocessing.LabelEncoder()
le.fit(label)
label = le.transform(label)
print(label)
print(data)