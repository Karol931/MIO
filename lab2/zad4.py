from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix


data = np.loadtxt("yeast.data", usecols = (1,2,3,4,5,6,7,8))
label = np.loadtxt("yeast.data", usecols = (9),dtype=str)
le = preprocessing.LabelEncoder()
le.fit(label)
label = le.transform(label)
act = ["identity", "logistic", "tanh", "relu"]
sol = [ "sgd", "adam"]
for s in sol:
    for a in act:
        data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=0.2, random_state=42, stratify=label)
        network = MLPClassifier(hidden_layer_sizes=(12, 8), max_iter=1000, activation=a,solver=s)
        network.fit(data_train, label_train)
        label_pred = network.predict(data_test)
        cm = confusion_matrix(label_test, label_pred)
        score = network.score(data_test,label_test)
        with open('zad4.txt', 'a') as f:
            f.write(" Activation: " + str(a) + " Solver: " + str(s) + "\n")
            f.write("Score = " + str(score) + "\n")
            f.write("Confusion matrix: \n" + str(cm) + "\n\n")
        print(str(score) + "\n")
        print(cm)
# print(label)
# print(data)