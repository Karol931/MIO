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

rain = ctrl.Antecedent(np.arange(0, 61, 1), 'rain')

rain["no"] = fuzz.trimf(rain.universe, [0, 5, 20])
rain["little"] = fuzz.trimf(rain.universe, [5, 20, 30])
rain["mid"] = fuzz.trimf(rain.universe, [20, 30, 40])
rain["alot"] = fuzz.trimf(rain.universe, [30, 40, 50])
rain["storm"] = fuzz.trimf(rain.universe, [40, 50, 60])

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


rule26 = ctrl.Rule(rain['no'] & temperature['cold'] & moisture['very_dry'], watering['alot'])
rule27 = ctrl.Rule(rain['no'] & temperature['mid'], watering['mid'])
rule28 = ctrl.Rule(rain['no'] & temperature['warm'] & moisture['moist'], watering['little'])
rule29 = ctrl.Rule(rain['little'] & temperature['hot'] & moisture['very_dry'], watering['max'])
rule30 = ctrl.Rule(rain['little'] & temperature['hot'] & moisture['dry'], watering['alot'])
rule31 = ctrl.Rule(rain['little'] & temperature['hot'] & moisture['mid'], watering['little'])
rule32 = ctrl.Rule(rain['mid'] & temperature['mid'], watering['dont'])
rule33 = ctrl.Rule(rain['alot'] & temperature['cold'] & moisture['very_moist'], watering['dont'])
rule34 = ctrl.Rule(rain['alot'] & temperature['mid'] & moisture['moist'], watering['dont'])
rule35 = ctrl.Rule(rain['storm'], watering['dont'])

control_system = ctrl.ControlSystem([rule1, rule2,rule3,rule4,rule5,rule6,rule7,rule8,rule9,rule10,rule11,rule12,rule13,rule14,rule15,rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25, rule26, rule27, rule28, rule29, rule30, rule31, rule32, rule33, rule34, rule35])


temperature_grid, moisture_grid = np.meshgrid(np.arange(15.1, 34.9, 0.1), np.arange(0.1, 99.9, 1))
test_points = np.transpose(np.vstack((np.ravel(temperature_grid),np.ravel(moisture_grid))))

rain_0 = np.array([15]*len(test_points[:,0]))
rain_1 = np.array([25]*len(test_points[:,0]))
rain_2 = np.array([35]*len(test_points[:,0]))
rain_3 = np.array([45]*len(test_points[:,0]))

rains = [('no', np.array([15]*len(test_points[:,0]))), ('little', np.array([25]*len(test_points[:,0]))), ('mid', np.array([35]*len(test_points[:,0]))), ('alot', np.array([45]*len(test_points[:,0]))), ('storm', np.array([55]*len(test_points[:,0])))]

rain.view()
plt.figure()

# for name, rain in rains: 
#     test_points = np.transpose(np.vstack((np.ravel(temperature_grid),np.ravel(moisture_grid))))

#     model = ctrl.ControlSystemSimulation(control_system)

#     model.input['temperature'] = test_points[:,0]
#     model.input['moisture'] = test_points[:,1]
#     model.input['rain'] = rain
#     model.compute()
#     print(model.output['watering'])
#     test_points = np.concatenate((test_points, model.output['watering'].reshape(-1,1)), axis=1)

#     sns.heatmap(pd.DataFrame(test_points, columns = ['temperature','moisture','watering']).pivot(index='moisture', columns='temperature', values='watering'), cmap = 'coolwarm')
#     plt.title(f"Heatmap for: {name}")
#     plt.figure()
# watering.view()
# temperature.view()
# moisture.view()



plt.show()

