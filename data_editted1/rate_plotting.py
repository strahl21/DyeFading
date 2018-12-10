import numpy as np
from mainFunc import *
from scipy.integrate import odeint
from scipy.optimize import minimize, curve_fit
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

def linear(x, a, b):
    return a * x + b

def get_K1(k_combined, K):
    OH_conc = 0.1
    return k_combined / (OH_conc * 1 / K)


#------------------------Simulate----------------------------------------------
"""
file_name = "120isothermalcsv.csv"
data = np.genfromtxt(file_name, skip_header = True, delimiter = ',')
# parse the data
time = data[:,0]
ph_abs = data[:,1]
k1_0 = 5.09
k2_0 = 0.87 #0.38
alpha_0 = 1
beta_0 = 1
result_simulate = np.array(rateint(k1_0, k2_0, alpha_0, beta_0, ph_abs[0], ph_abs[0], time)[:,-1])
print((sum(result_simulate ** 2 - ph_abs** 2)))
plt.scatter(time, ph_abs)
plt.plot(time, result_simulate)
plt.show()
"""

#data = np.genfromtxt("data.csv", skip_header = True, delimiter = ',')
# for test
file_names = ["60isothermal.csv", "90isothermal.csv", "120isothermalcsv.csv"]
temperature = [288.706, 305.372, 322.039]
k1_0 = [0.2, 0.3, 0.6]
k2_0 = [0.03, 0.01, 0.065]

alpha = np.ones(len(temperature)) * 1.0
beta = np.ones(len(temperature)) * 1.0
k1_result = []
k2_result = []
data_begin = [9, 10, 9]
data_limit = [1100, 450, 300]
K_result = []

"""
plt.figure()
for i in range(len(file_names)):
    data = np.genfromtxt(file_names[i], skip_header = True, delimiter = ',')
    time = data[:,0]#[data_begin[i]:data_limit[i]]
    ph_abs = data[:,1]#[data_begin[i]:data_limit[i]]
    maxAbs = ph_abs[0]
    T_equil = np.ones(len(ph_abs)) * ph_abs[-1]
    K_equil = (ph_abs[0] - ph_abs[-1]) / ph_abs[-1]
    for j in range(len(ph_abs)):
        if (T_equil[j] >= (ph_abs[j] - 0.09)):
            time = time[:j]
            ph_abs = ph_abs[:j]
            T_equil = T_equil[:j]
            break;

# *** temperature must be in Kelvin ***
# parse the data
    array_params, covariance = curve_fit(linear, time, np.log(ph_abs))
    slope, intercept = array_params
    residuals = np.log(ph_abs)- np.array([linear(i, slope, intercept) for i in time])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.log(ph_abs) - np.mean(np.log(ph_abs))) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("k1 = ", slope)
    #print([np.log(linear(i, slope, intercept)) for i in time])
    plt.plot(time, [(linear(i, slope, intercept)) for i in time], color = 'k', label = "curve %d linear fit %0.2f $R^2$ value" %(i + 1, r_squared), linewidth = 2.0)
    plt.plot(time, np.log(ph_abs), label = 'curve %d' %(i + 1))
plt.legend()
plt.ylim([-1.5, -0.8])
plt.xlim(0, 20)
plt.show()
"""
plt.figure()
for i in range(len(file_names)):
    data = np.genfromtxt(file_names[i], skip_header = True, delimiter = ',')
# *** temperature must be in Kelvin ***
# parse the data
    time = data[:,0][data_begin[i]:]
    ph_abs = data[:,1][data_begin[i]:]
    maxAbs = ph_abs[0]
    T_equil = np.ones(len(ph_abs)) * ph_abs[-1]
    K_equil = (ph_abs[0] - ph_abs[-1]) / ((ph_abs[-1]) * 0.1)
    K_result.append(K_equil)

    for j in range(len(ph_abs)):
        if (T_equil[j] >= (ph_abs[j] - 0.09)):
            time = time[:j]
            ph_abs = ph_abs[:j]
            T_equil = T_equil[:j]
            break;

    array_params, covariance = curve_fit(linear, time, np.log(np.log(ph_abs)-np.log(T_equil)))
    slope, intercept = array_params
    #K_equil = ((intercept) - np.log(ph_abs[-1])) / (np.log(ph_abs[-1]) * 0.1)
    residuals = np.log(np.log(ph_abs)-np.log(T_equil)) - np.array([linear(i, slope, intercept) for i in time])
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((np.log(np.log(ph_abs)-np.log(T_equil)) - np.mean(np.log(np.log(ph_abs)-np.log(T_equil)))) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print("combined_k = ", slope)
    print("Kequil = ", K_equil)
    k1 = get_K1(-1 * slope, K_equil)
    print("k1 = ", k1 / 60)
    print("k-1 = ",(k1 / K_equil) / 60)
    #print([np.log(linear(i, slope, intercept)) for i in time])
    if i > 1:
        end = 3
    else:
        end = 2

    plt.plot(time, [(linear(i, slope, intercept)) for i in time], color = 'k', label = "Temp %s F linear fit %0.2f $R^2$ value" %(file_names[i][0:end], r_squared), linewidth = 2.0)
    plt.plot(time, (np.log(np.log(ph_abs)-np.log(T_equil))), label = 'Temp %s F' %(file_names[i][0:end]))
    plt.xlabel('time (min)')
    plt.ylabel('$ln(A - A_e)$')
plt.legend()
#plt.ylim([-1.5, -0.8])
#plt.xlim(0, 20)
plt.show()

print(K_result)

plt.figure()
plt.plot(np.array(temperature[1:]) ** -1, np.log(np.array(K_result[1:])))
plt.xlabel("1 / T")
plt.ylabel("$ln(K)$")
plt.show()

array_params, covariance = curve_fit(linear, np.array(temperature[1:]) ** -1, np.log(np.array(K_result[1:])))
slope, intercept = array_params
print(slope * -1 * 8.3145)





"""
# initialization
    method = 'SLSQP'
    initialization = np.array([k1_0[i], k2_0[i]])
    bounds = ((0,1), (0,0.01))
    print("Initial sum of squared error = ", objective_function_dye(initialization, ph_abs, maxAbs, alpha[i], beta[i], ph_abs[0], time))

    #bounds = ((0, 0.3), (0, 1), (0.4, 0.7), (0.1, 0.5))
    #measured, maxAbs, alpha, beta, initialval, time
    regression_result = minimize(objective_function_dye, initialization, args=(ph_abs, maxAbs, alpha[i], beta[i], ph_abs[0], time), bounds = bounds, method=method)#, bounds=bounds)

    if regression_result.success:
        print("Success")
    else:
        print(regression_result.message)

    k1, k2 = regression_result.x

    print("Final sum of squared error = ", objective_function_dye([k1, k2], ph_abs, maxAbs, alpha[i], beta[i], ph_abs[0], time))

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
