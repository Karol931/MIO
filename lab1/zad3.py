from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt

data = load_iris().data
label = load_iris().target
#print(label)
#print(label_test)
neuron = Perceptron(tol=1e-3, max_iter=20)
for i in range(5):
    train, test, label_train, label_test = train_test_split(data,label,test_size=0.2,train_size=0.8)
    neuron.fit(train,label_train)
    print("Score: " + str(neuron.score(test, label_test)))
    predicted = neuron.predict(test)
    conf_matrix = confusion_matrix(label_test, predicted)
    print("Coffusion matrix:\n" + str(conf_matrix))
