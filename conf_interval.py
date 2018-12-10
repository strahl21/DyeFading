import numpy as np
from mainFunc import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#data = np.genfromtxt("data.csv", skip_header = True, delimiter = ',')
# for test
file_name = "completely_matched_2_runs.csv"
data = np.genfromtxt(file_name, skip_header = True, delimiter = ',')
# *** temperature must be in Kelvin ***
# parse the data
time = data[:,0]
temperature = data[:,2]
ph_abs = data[:,1]
temperature_2 = data[:,4]
ph_abs_2 = data[:,3]

def arrhenius(ea,a,T):
    r = 8.314
    return a * np.exp (-ea / (r*T))

def rate(ea,a,T,conc,power):
    return arrhenius(ea,a,T) * conc ** power

def rateph2minus(ph2minus,t,ea1,ea2,a1,a2,T,alpha,beta):
    ph3minus = 1.145565033 - ph2minus
    #ph3minus =  0.994207 - ph2minus

    return rate(ea2,a2,T,ph3minus,beta)-rate(ea1,a1,T,ph2minus,alpha)

def rateint(ea1,ea2,a1,a2,T,alpha,beta,initialval, time):
    return odeint(rateph2minus, initialval, time, args=(ea1,ea2,a1,a2,T,alpha,beta))[1]

E_a_1 =  54164.24
E_a_2 =  80650.21
A_1 =  3504786.0
A_2 =  10864850000.0
alpha =  1.0
beta =  1.0


def loop(E_a_1, E_a_2, A_1, A_2, temperature, alpha, beta, time):
    result = []
    initialval = ph_abs[0]
    for i in range(1, len(time)):
        newtime = time[i] - time[i-1]
        timeint = np.array([0,newtime])
        #print(timeint)
        T = temperature[i]
        initialval2 = rateint(E_a_1, E_a_2, A_1, A_2, T, alpha, beta, initialval, timeint)[0]
        #print(initialval2)
        result.append(initialval2)
        initialval = initialval2
    return result

result2 = loop(E_a_1, E_a_2, A_1, A_2, temperature, alpha, beta, time)
#print(result2)
plt.plot(time[1:], result2)
plt.show()

#a1,a2,T,alpha,beta,initialval, time
a1_array = np.ones(len(ph_abs[1:])) * A_1
a2_array = np.ones(len(ph_abs[1:])) * A_2
alpha_array = np.ones(len(ph_abs[1:])) * alpha
beta_array = np.ones(len(ph_abs[1:])) * beta


A_1_array = [A_1, A_1 * 0.95, A_1 + A_1 * 0.05]
A_2_array = [A_2, A_2 * 0.95, A_2 + A_2 * 0.05]
perturb1 = 0.005
perturb2 = 0.005
numPoints1 = 20
numPoints2 = 20
results_contour = []
label_plot = []
param1Range = np.linspace(E_a_1 * (1 - perturb1), E_a_1 * (1 + perturb1), numPoints1)
param2Range = np.linspace(E_a_2 * (1 - perturb2), E_a_2 * (1 + perturb2), numPoints2)
param1_grid, param2_grid = np.meshgrid(param1Range, param2Range)


for i in A_1_array:
    for j in A_2_array:
        args = (i,j,temperature,alpha,beta,time)
        results_contour.append(nonLinearConfInt(loop, E_a_1, E_a_2, ph_abs[1:], args = args, show = False, perturb1 = perturb1, perturb2 = perturb2, pts1=numPoints1, pts2=numPoints2, title = "A_1 = %.3f A_2 = %.3f "%(i, j)))
        label_plot.append("A_1 = %.3f A_2 = %.3f "%(i, j))

plt.figure()
for i in range(len(results_contour)):
    xContourValues = np.array(results_contour[i].vertices[:,0])
    xContourValues = np.append(xContourValues, xContourValues[0])
    yContourValues = np.array(results_contour[i].vertices[:,1])
    yContourValues = np.append(yContourValues, yContourValues[0])
    plt.plot(xContourValues, yContourValues, label = label_plot[i], linewidth = 1.5)

plt.title('Confidence regions')
plt.xlabel('E_a_1')
plt.ylabel('E_a_2')
plt.legend()
plt.show()
