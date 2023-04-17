import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

df = pd.read_csv("./dane_pogodowe.csv")

temperature = ctrl.Antecedent(np.arange(-10,31,0.1),'temperature')

temperature['cold'] = fuzz.gaussmf(temperature.universe, -10, 5)
temperature['mid'] = fuzz.gaussmf(temperature.universe, 0,5)
temperature['warm'] = fuzz.gaussmf(temperature.universe, 10,5)
temperature['hot'] = fuzz.gaussmf(temperature.universe, 20, 5)
temperature['very_hot'] = fuzz.gaussmf(temperature.universe, 30, 5)

moisture = ctrl.Antecedent(np.arange(15, 86, 1),'moisture')

moisture['dry'] = fuzz.gaussmf(moisture.universe, 15,10)
moisture['mid'] = fuzz.gaussmf(moisture.universe, 50,15)
moisture['moist'] = fuzz.gaussmf(moisture.universe, 85,10)

rain = ctrl.Antecedent(np.arange(0, 81, 1), 'rain')

rain["no"] = fuzz.gaussmf(rain.universe, 0,5)
rain["mid"] = fuzz.gaussmf(rain.universe, 20,30)
rain["storm"] = fuzz.gaussmf(rain.universe, 80, 60)

wind_speed = ctrl.Antecedent(np.arange(0, 11), 'wind_speed')

wind_speed['calm'] = fuzz.gaussmf(wind_speed.universe, 0, 5)
wind_speed['mid'] = fuzz.gaussmf(wind_speed.universe, 4, 5)
wind_speed['windy'] = fuzz.gaussmf(wind_speed.universe, 10, 5)

pm10 = ctrl.Consequent(np.arange(0, 121), 'pm10')

pm10['low'] = fuzz.gaussmf(pm10.universe, 0, 20)
pm10['mid'] = fuzz.gaussmf(pm10.universe, 30, 20)
pm10['high'] = fuzz.gaussmf(pm10.universe, 60, 20)
pm10['very_high'] = fuzz.gaussmf(pm10.universe, 120, 30)


rule1 = ctrl.Rule(temperature['cold'] & moisture['dry'], pm10['low'])
rule2 = ctrl.Rule(temperature['mid'] & (moisture['mid'] | moisture['moist']) & (rain['no'] | rain['mid']), pm10['mid'])
rule3 = ctrl.Rule((temperature['hot'] | temperature['very_hot']) & (wind_speed['calm'] | wind_speed['mid']) & moisture['mid'] & rain['no'], pm10['low'])
rule4 = ctrl.Rule((temperature['warm'] | temperature['hot']) & wind_speed['windy'] & moisture['mid'] & rain['no'], pm10['mid'])
rule5 = ctrl.Rule((temperature['warm'] | temperature['hot']) & wind_speed['windy'] & (moisture['moist'] | rain['storm']), pm10['high'])
rule6 = ctrl.Rule(temperature['very_hot'] & moisture['dry'], pm10['very_high'])
rule7 = ctrl.Rule(temperature['very_hot'] & (moisture['mid'] | moisture['moist']) & (rain['no'] | rain['mid']), pm10['high'])
rule8 = ctrl.Rule(temperature['very_hot'] & (moisture['moist'] | rain['storm']), pm10['very_high'])

control_system = ctrl.ControlSystem([rule1, rule2,rule3,rule4,rule5,rule6,rule7,rule8])

model = ctrl.ControlSystemSimulation(control_system)

model.input['temperature'] = np.array(df['averageAirTemp'])
model.input['moisture'] = np.array(df['averageRelativeHumidity'])
model.input['wind_speed'] = np.array(df['averageWindSpeed'])
model.input['rain'] = np.array(df['rainAccumulation'])
model.compute()

calculeted = model.output['pm10']
expected = df["averagePm10"]
print(r2_score(calculeted, expected))

temperature.view()
wind_speed.view()
moisture.view()
rain.view()
pm10.view()



plt.show()

