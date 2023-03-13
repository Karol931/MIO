from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt

matrix = np.loadtxt("fuel.txt", dtype=float)
neuron = Perceptron(tol=1e-3)
neuron2 = Perceptron(max_iter=5)
#print(matrix)

neuron.fit(matrix[:,0:2],matrix[:,3])
print(neuron.score(matrix[:,0:2],matrix[:,3]))

neuron2.fit(matrix[:,0:2],matrix[:,3])
print(neuron2.score(matrix[:,0:2],matrix[:,3]))

