import skfuzzy
import pandas as pd
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("planets.csv", skiprows=1)
data = df.iloc[:,1:6]

scaler = StandardScaler()
data = scaler.fit_transform(data)

center,_,_,_,_,_,fpc = skfuzzy.cluster.cmeans(data.T, 5, 2, error=1e-5, maxiter=1e6)

print("Centers: ")
for i in range(5):
    print("Center " + str(i) + ": " + str(center[i]))

print("FPC: " + str(fpc))