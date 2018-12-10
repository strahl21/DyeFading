import numpy as np
from mainFunc import *
from scipy.integrate import odeint
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def rate(k, conc, power):
    return k * conc ** power

def rateph2minus(ph2minus, t, k1, k2, alpha, beta, maxAbs):
    ph3minus = maxAbs - ph2minus
    return rate(k2, ph3minus, beta) - rate(k1, ph2minus, alpha)

def rateint(k1, k2, alpha, beta, maxAbs, initialval, time):
    return odeint(rateph2minus, initialval, time, args=(k1, k2 ,alpha, beta, maxAbs))

def objective_function_dye(vars, measured, maxAbs, alpha, beta, initialval, time):
    predicted_result = rateint(vars[0], vars[1], alpha, beta, maxAbs, initialval, time)[:,-1]
    sum_sq_errors = (sum(abs(predicted_result ** 2 - measured ** 2)))
    return sum_sq_errors

#------------------------Simulate----------------------------------------------

file_name = "90isothermal.csv"
data = np.genfromtxt(file_name, skip_header = True, delimiter = ',')
# parse the data
time = data[:,0]
ph_abs = data[:,1]
k1_0 = 0.2
k2_0 = 0.01
alpha_0 = 1
beta_0 = 1
result_simulate = np.array(rateint(k1_0, k2_0, alpha_0, beta_0, 1.323092699, ph_abs[0], time)[:,-1])
print((sum(result_simulate ** 2 - ph_abs** 2)))
plt.scatter(time, ph_abs)
plt.plot(time, result_simulate)
plt.show()

"""
#data = np.genfromtxt("data.csv", skip_header = True, delimiter = ',')
# for test
file_names = ["60isothermal.csv", "90isothermal.csv", "120isothermal.csv"]
temperature = [288.706, 305.372, 322.039]
maxAbs = [1.038394094, 1.04479754, 1.323092699]
k1_0 = [0.06, 0.08, 0.6]
k2_0 = [0.0065, 0.001, 0.065]

alpha = np.ones(len(temperature)) * 1.0
beta = np.ones(len(temperature)) * 1.0
k1_result = []
k2_result = []

for i in range(len(file_names)):
    data = np.genfromtxt(file_names[i], skip_header = True, delimiter = ',')
# *** temperature must be in Kelvin ***
# parse the data
    time = data[:,0]
    ph_abs = data[:,1]

# initialization
    method = 'SLSQP'
    initialization = np.array([k1_0[i], k2_0[i]])
    print("Initial sum of squared error = ", objective_function_dye(initialization, ph_abs, maxAbs[i], alpha[i], beta[i], ph_abs[0], time))

    #bounds = ((0, 0.3), (0, 1), (0.4, 0.7), (0.1, 0.5))
    #measured, maxAbs, alpha, beta, initialval, time
    regression_result = minimize(objective_function_dye, initialization, args=(ph_abs, maxAbs[i], alpha[i], beta[i], ph_abs[0], time), method=method)#, bounds=bounds)

    print("Initial sum of squared error = ", objective_function_dye(initialization, ph_abs, maxAbs[i], alpha[i], beta[i], ph_abs[0], time))

    if regression_result.success:
        print("Success")
    else:
        print(regression_result.message)

    k1, k2 = regression_result.x

    print("Final sum of squared error = ", objective_function_dye([k1, k2], ph_abs, maxAbs[i], alpha[i], beta[i], ph_abs[0], time))

    print("k1 = ", k1)
    print("k2 = ", k2)
    k1_result.append(k1)
    k2_result.append(k2)

temperature_plot = 1 / np.array(temperature)
plt.plot(temperature_plot, np.log(np.array(k1_result)))
plt.scatter(temperature_plot, np.log(np.array(k1_result)))
plt.show()

plt.plot(temperature_plot, np.log(np.array(k2_result)))
plt.scatter(temperature_plot, np.log(np.array(k2_result)))
plt.show()
"""
