from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from matplotlib import pyplot as plt

data = load_iris().data
label = load_iris().target
iterations = [1,2,5,10,20,40,80,200,500,1000]
accuracy = [0,0,0,0,0,0,0,0,0,0]
counter = 0
for c in range(10):
    train, test, label_train, label_test = train_test_split(data,label,test_size=0.2,train_size=0.8)
    for i in iterations:
        neuron = Perceptron(early_stopping = False, max_iter=i)
        neuron.fit(train,label_train)
        accuracy[counter] += neuron.score(test, label_test)
        counter = counter+1
    counter = 0

for i in range(len(accuracy)):
    accuracy[i] /= 10
    print(accuracy[i])

plt.xscale('log')
plt.scatter(iterations,accuracy)
plt.xlabel('Number of iterations')
plt.ylabel('Average accuracy')
plt.show()