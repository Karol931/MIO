import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import seaborn as sns
import pandas as pd

temperature = ctrl.Antecedent(np.arange(15, 36, 0.1),'temperature')

temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 20])
temperature['mid'] = fuzz.trimf(temperature.universe, [16, 20, 23])
temperature['warm'] = fuzz.trimf(temperature.universe, [20, 23, 26])
temperature['hot'] = fuzz.trimf(temperature.universe, [23, 26, 30])
temperature['very_hot'] = fuzz.trimf(temperature.universe, [26, 30, 35])

moisture = ctrl.Antecedent(np.arange(0, 101, 1),'moisture')

moisture['very_dry'] = fuzz.trimf(moisture.universe, [0, 0, 30])
moisture['dry'] = fuzz.trimf(moisture.universe, [0, 30, 50])
moisture['mid'] = fuzz.trimf(moisture.universe, [30, 50, 75])
moisture['moist'] = fuzz.trimf(moisture.universe, [50, 75, 100])
moisture['very_moist'] = fuzz.trimf(moisture.universe, [75, 100, 100])

watering = ctrl.Consequent(np.arange(0, 26, 0.1),'watering')


watering['dont'] = fuzz.trimf(watering.universe, [0, 0, 1])
watering['little'] = fuzz.trimf(watering.universe, [0, 5, 9])
watering['mid'] = fuzz.trimf(watering.universe, [5, 9, 12])
watering['alot'] = fuzz.trimf(watering.universe, [9, 12, 19])
watering['max'] = fuzz.trimf(watering.universe, [12, 19, 25])

rule1 = ctrl.Rule(temperature['cold'] & moisture['very_dry'],watering['alot']) 
rule2 = ctrl.Rule(temperature['cold'] & moisture['dry'],watering['mid']) 
rule3 = ctrl.Rule(temperature['cold'] & moisture['mid'],watering['little']) 
rule4 = ctrl.Rule(temperature['cold'] & moisture['moist'],watering['dont']) 
rule5 = ctrl.Rule(temperature['cold'] & moisture['very_moist'],watering['dont']) 

rule6 = ctrl.Rule(temperature['mid'] & moisture['very_dry'],watering['alot']) 
rule7 = ctrl.Rule(temperature['mid'] & moisture['dry'],watering['mid']) 
rule8 = ctrl.Rule(temperature['mid'] & moisture['mid'],watering['little']) 
rule9 = ctrl.Rule(temperature['mid'] & moisture['dry'],watering['mid']) 
rule10 = ctrl.Rule(temperature['mid'] & moisture['very_moist'],watering['little']) 

rule11 = ctrl.Rule(temperature['warm'] & moisture['very_dry'],watering['alot']) 
rule12 = ctrl.Rule(temperature['warm'] & moisture['dry'],watering['alot']) 
rule13 = ctrl.Rule(temperature['warm'] & moisture['mid'],watering['mid']) 
rule14 = ctrl.Rule(temperature['warm'] & moisture['dry'],watering['little']) 
rule15 = ctrl.Rule(temperature['warm'] & moisture['very_moist'],watering['dont']) 

rule16 = ctrl.Rule(temperature['hot'] & moisture['very_dry'],watering['max']) 
rule17 = ctrl.Rule(temperature['hot'] & moisture['dry'],watering['alot']) 
rule18 = ctrl.Rule(temperature['hot'] & moisture['mid'],watering['alot']) 
rule19 = ctrl.Rule(temperature['hot'] & moisture['dry'],watering['little']) 
rule20 = ctrl.Rule(temperature['hot'] & moisture['very_moist'],watering['little']) 

rule21 = ctrl.Rule(temperature['very_hot'] & moisture['very_dry'],watering['max']) 
rule22 = ctrl.Rule(temperature['very_hot'] & moisture['dry'],watering['max']) 
rule23 = ctrl.Rule(temperature['very_hot'] & moisture['mid'],watering['alot']) 
rule24 = ctrl.Rule(temperature['very_hot'] & moisture['moist'],watering['mid']) 
rule25 = ctrl.Rule(temperature['very_hot'] & moisture['very_moist'],watering['little']) 


control_system = ctrl.ControlSystem([rule1, rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25])

model = ctrl.ControlSystemSimulation(control_system)

temperature_grid, moisture_grid = np.meshgrid(np.arange(15.1, 34.9, 0.1), np.arange(0.1, 99.9, 1))
test_points = np.transpose(np.vstack((np.ravel(temperature_grid),np.ravel(moisture_grid))))


model.input['temperature'] = test_points[:,0]
model.input['moisture'] = test_points[:,1]
model.compute()
print(model.output['watering'])
test_points = np.concatenate((test_points, model.output['watering'].reshape(-1,1)), axis=1)

sns.heatmap(pd.DataFrame(test_points, columns = ['temperature','moisture','watering']).pivot(index='moisture', columns='temperature', values='watering'), cmap = 'coolwarm')
watering.view()
temperature.view()
moisture.view()


plt.show()

