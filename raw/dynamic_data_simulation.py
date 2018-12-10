from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

#data = np.genfromtxt("data.csv", skip_header = True, delimiter = ',')
# for test
file_name = "completely_matched.csv"
data = np.genfromtxt(file_name, skip_header = True, delimiter = ',')
# *** temperature must be in Kelvin ***
# parse the data
time = data[:,0]
temperature = data[:,2]
ph_abs = data[:,1]

# initial guesses from literature
E_a_1_initial_guess =  35940.56     # J / mol
E_a_2_initial_guess =  77278       # J / mol
A_1_initial_guess = 1.14e4          # m ^ 3 / mol s
A_2_initial_guess = 3.06e9           # 1 / s4
alpha_initial_guess = 1            # reaction order
beta_initial_guess = 1             # reaction order
Ph_2_minus_initial = 0.461204797    # get from data
Ph_3_minus_initial = 1.145565033 - ph_abs[0]

array_of_end_values = []
def solve(var):
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

    E_a_1 = reaction_model.FV(E_a_1_initial_guess)
    E_a_2 = reaction_model.FV(E_a_2_initial_guess)
    A_1 = reaction_model.FV(A_1_initial_guess)
    A_2 = reaction_model.FV(A_2_initial_guess)
    alpha = reaction_model.FV(alpha_initial_guess)
    beta = reaction_model.FV(beta_initial_guess)

    # NO bounds
    # state Variables
    Ph_3_minus = reaction_model.SV(Ph_3_minus_initial)

    # variable we will use to regress other Parameters
    Ph_2_minus = reaction_model.SV(Ph_2_minus_initial)

    # intermediates
    k1 = reaction_model.Intermediate(A_1 * reaction_model.exp(-E_a_1 / (R * T)))
    k2 = reaction_model.Intermediate(A_2 * reaction_model.exp(-E_a_2 / (R * T)))
    r1 = reaction_model.Intermediate(k1 * Ph_2_minus ** alpha)
    r2 = reaction_model.Intermediate(k2 * Ph_3_minus ** beta)

    # equations
    reaction_model.Equations([Ph_2_minus.dt() == r2 - r1,
                          Ph_3_minus.dt() == r1 - r2])

    # parameter options
    # controlled variable options

    # model options and other to solve
    reaction_model.options.IMODE = 4        # set up dynamic simulation
    reaction_model.options.NODES = 2        # number of nodes for collocation equations
    reaction_model.options.SOLVER = 1       # use APOPT active set non-linear solverreaction_model.options.EV_TYPE = 2      # use l-1 norm rather than 2 norm

    reaction_model.solve(disp=True)
    return Ph_2_minus.value

variable = [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4]

#for i in variable:
plt.plot(time, solve(10), linewidth = 3, color = "k")
plt.scatter(time, ph_abs)
#plt.plot(reaction_model.time, Ph_3_minus.value)
plt.xlabel("Time")
plt.ylabel("Ph 2 minus")
plt.show()
