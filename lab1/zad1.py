from sklearn.linear_model import Perceptron
import numpy as np
from matplotlib import pyplot as plt

n = [5,10,20,100]

for i in n: 
    k1 = np.concatenate((np.random.normal([0,-1],[1,1],[i,2]),np.random.normal([1,1],[1,1],[i,2])))
    k1_test = np.concatenate((np.random.normal([0,-1],[1,1],[200,2]),np.random.normal([1,1],[1,1],[200,2])))
    x = np.concatenate(([0]*i, [1]*i))
    x_test = np.concatenate(([0]*200, [1]*200))
    neuron = Perceptron(tol=1e-3, max_iter=20)

    #plt.scatter(k1[:,0],k1[:,1],c=x)

    neuron.fit(k1,x)
    print(neuron.score(k1_test,x_test))
    x1 = np.linspace(-10,10,100)
    x2 = -(1./neuron.coef_[0][1])*(neuron.coef_[0][0]*x1+neuron.intercept_[0])

    plt.figure(i)
    plt.plot(x1,x2,'-r')
    colors = np.concatenate((["red"]*200,["green"]*200))
    plt.scatter(np.array(k1_test)[:,0], np.array(k1_test)[:,1], c=colors)
    plt.scatter(np.array(k1)[:,0], np.array(k1)[:,1], c=x)
plt.show()

print(k1)