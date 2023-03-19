from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

data = np.loadtxt("treatment.txt", delimiter=",")
#print(data)

label = data[:,2]
data = data[:,[0,1]]
#print(label)
#print(data)
data[:,0] /= np.max(data[:,0])
data[:,1] /= np.max(data[:,1])
arr = [[7, 29, 40, 9],[24, 22, 15],[10, 18, 22],[37, 6, 39],[10, 17, 31],[27, 15],[5, 29, 31],[14, 38, 5, 36]]
for i in range(8):
    data_train, data_test, label_train, label_test = train_test_split(data,label,stratify = label,test_size=0.2,train_size=0.8)
    network = MLPClassifier(hidden_layer_sizes=(arr[i]), max_iter = 1000, tol = 0.001)
    network.fit(data_train,label_train)
    predicted_labels = network.predict(data_train)
    score = network.score(data_test,label_test)
    print(score)
    xx, yy = np.meshgrid(np.arange(0, 1, 0.025), np.arange(0, 1, 0.025))
    test_points = np.transpose(np.vstack((np.ravel(xx),np.ravel(yy))))
    prediction = network.predict(test_points)
    plt.figure(i)
    plt.scatter(test_points[:,0], test_points[:,1], c=prediction)
plt.show()