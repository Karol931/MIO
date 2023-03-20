from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits

data = load_digits().data
label = load_digits().target
# print(data)
# print(label)
sol = [ "sgd", "adam"]
l_rate = ["constant", "invscaling", "adaptive"]
act = ["identity", "logistic", "tanh", "relu"]
arr = [[7, 29, 40, 9],[24, 22, 15],[10, 18, 22]]
iter = [20,50,100,1000]
counter = 0
for ar in arr:
    for s in sol:
        for ac in act:
            for i in iter:
                data_train, data_test, label_train, label_test = train_test_split(data,label,stratify = label,test_size=0.2,train_size=0.8)
                if s == "sgd":
                    for l in l_rate:
                        network = MLPClassifier(hidden_layer_sizes=(ar), max_iter = i, tol = 0.001, learning_rate=l,activation=ac,solver=s)
                else:
                    network = MLPClassifier(hidden_layer_sizes=(ar), max_iter = i, tol = 0.001, activation=ac,solver=s)
                network.fit(data_train,label_train)
                predicted_labels = network.predict(data_train)
                score = network.score(data_test,label_test)
                if s == "sgd":
                    for l in l_rate:
                        with open('zad3.txt', 'a') as f:
                            f.write("Network: " + str(ar) +" Max_iter: " + str(i) + " Activation: " + str(ac) + " Solver: " + str(s) + " l_rate: " + str(l) + "\n")
                            f.write("Score = " + str(score) + "\n")
                else:
                    with open('zad3.txt', 'a') as f:
                        f.write("Network: " + str(ar) +" Max_iter: " + str(i) + " Activation: " + str(ac) + " Solver: " + str(s) +  "\n")
                        f.write("Score = " + str(score) + "\n")
                counter = counter + 1
print(counter)