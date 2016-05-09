import skfuzzy as fuzz
import numpy as np
import matplotlib.pyplot as plt
# New Antecedent/Consequent objects hold universe variables and membership
# functions
metric_range = np.arange(0,1,0.000001)
cpu = fuzz.Antecedent(metric_range, 'cpu')
mem_usage = fuzz.Antecedent(metric_range, 'mem_usage')
tip = fuzz.Consequent(np.arange(0, 26, 1), 'tip')

# Auto-membership function population is possible with .automf(3, 5, or 7)
# cpu['poor'] = fuzz.trimf(cpu.universe, [0, 0, 3])
# cpu['average'] = fuzz.trimf(cpu.universe, [0, 3, 7])
# cpu['good'] = fuzz.trimf(cpu.universe, [7, 11, 11])
cpu.automf(3)
mem_usage.automf(3)

# Custom membership functions can be built interactively with a familiar,
# Pythonic API
tip['low'] = fuzz.trimf(tip.universe, [0, 0, 13])
tip['medium'] = fuzz.trimf(tip.universe, [0, 13, 25])
tip['high'] = fuzz.trimf(tip.universe, [13, 25, 25])
cpu_t = cpu
# You can see how these look with .view()
ax = cpu.view()

rule1 = fuzz.Rule(cpu['poor'] & cpu['average'] & mem_usage['poor'], tip['low'])
rule2 = fuzz.Rule(mem_usage['average'], tip['medium'])
rule3 = fuzz.Rule(mem_usage['good'] & cpu['good'], tip['high'])

autoscaling = fuzz.ControlSystemSimulation(fuzz.ControlSystem(rules=[rule1,rule2,rule3]))
autoscaling.input['cpu']=0.8
autoscaling.input['mem_usage']=0.012
autoscaling.compute()
print autoscaling.output['tip']
ax = tip.view(sim=autoscaling)
plt.show()