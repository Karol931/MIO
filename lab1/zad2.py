from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt

matrix = np.loadtxt("fuel.txt", dtype=float)
neuron = Perceptron(tol=1e-3, max_iter=20)
#print(matrix)
for i in range(5): 
    neuron.fit(matrix[:,0:2],matrix[:,3])
    print(neuron.score(matrix[:,0:2],matrix[:,3]))

