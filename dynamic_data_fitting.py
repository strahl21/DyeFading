from gekko import GEKKO
import numpy as np
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

# initial guesses from literature
E_a_1_initial_guess =  35940.56     # J / mol
E_a_2_initial_guess =  77278       # J / mol
A_1_initial_guess = 0.114e6          # m ^ 3 / mol s
A_2_initial_guess = 3.06e9           # 1 / s4
alpha_initial_guess = 1            # reaction order
beta_initial_guess = 1             # reaction order
Ph_2_minus_initial = ph_abs[0]    # get from data
#Ph_3_minus_initial = 1.145565033 - ph_abs[0]
Ph_3_minus_initial = 0.994207 - ph_abs[0]

# initialize model
reaction_model = GEKKO()

# set model time as time gathered data
reaction_model.time = time

# Constants
R = reaction_model.Const(8.3145)        # Gas constant J / mol K

# Parameters
T = reaction_model.Param(temperature)

# Fixed Variables to change
# bounds
"""
E_a_1 = reaction_model.FV(E_a_1_initial_guess, lb = 33900, ub = 37900)
E_a_2 = reaction_model.FV(E_a_2_initial_guess, lb = 73000, ub = 81000)
A_1 = reaction_model.FV(A_1_initial_guess, lb = 0)
A_2 = reaction_model.FV(A_2_initial_guess, lb = 0)
alpha = reaction_model.FV(alpha_initial_guess)#, lb = 0, ub = 1)
beta = reaction_model.FV(beta_initial_guess)#, lb = 0, ub = 1)
"""

# NO bounds

E_a_1 = reaction_model.FV(E_a_1_initial_guess)
E_a_2 = reaction_model.FV(E_a_2_initial_guess)
A_1 = reaction_model.FV(A_1_initial_guess)
A_2 = reaction_model.FV(A_2_initial_guess)
alpha = reaction_model.FV(alpha_initial_guess)
beta = reaction_model.FV(beta_initial_guess)

# one-sided bounds
#alpha.LOWER = 0
#beta.LOWER = 0

# state Variables
Ph_3_minus = reaction_model.SV(Ph_3_minus_initial)

# variable we will use to regress other Parameters
Ph_2_minus = reaction_model.CV(ph_abs)

# intermediates
k1 = reaction_model.Intermediate(A_1 * reaction_model.exp(-E_a_1 / (R * T)))
k2 = reaction_model.Intermediate(A_2 * reaction_model.exp(-E_a_2 / (R * T)))
# forward reaction
r1 = reaction_model.Intermediate(k1 * Ph_2_minus ** alpha)
# backwards reaction
r2 = reaction_model.Intermediate(k2 * Ph_3_minus ** beta)

# equations
reaction_model.Equations([Ph_2_minus.dt() == r2 - r1,
                          Ph_3_minus.dt() == r1 - r2])

# parameter options
E_a_1.STATUS = 1
E_a_2.STATUS = 1
A_1.STATUS = 0
A_2.STATUS = 0
alpha.STATUS = 0
beta.STATUS = 0

# controlled variable options
Ph_2_minus.MEAS_GAP = 1e-3
Ph_2_minus.STATUS = 1           # regress to this value
Ph_2_minus.FSTATUS = 1          # take in data measurements

# model options and other to solve
reaction_model.options.IMODE = 5         # set up dynamic estimation
reaction_model.options.NODES = 2        # number of nodes for collocation equations
reaction_model.options.SOLVER = 1       # use APOPT active set non-linear solver
reaction_model.options.EV_TYPE = 2      # use l-1 norm rather than 2 norm

reaction_model.solve(disp=True)

print("E_a 1 = ", E_a_1.value[-1])
print("E_a 2 = ", E_a_2.value[-1])
print("A_1 = ", A_1.value[-1])
print("A_2 = ", A_2.value[-1])
print("alpha = ", alpha.value[-1])
print("beta = ", beta.value[-1])
#print("ph 2 minus = ", Ph_2_minus.value)

plt.plot(reaction_model.time, Ph_2_minus.value, linewidth = 3, color = "k")
plt.scatter(reaction_model.time, ph_abs)
plt.plot(reaction_model.time, Ph_3_minus.value)
plt.xlabel("Time")
plt.ylabel("Ph 2 minus")
plt.show()

plt.plot(reaction_model.time, T.value)
plt.xlabel("Time")
plt.ylabel("Temperature")
plt.show()
