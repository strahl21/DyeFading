import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import math

def nonLinearConfInt(func, param1, param2, dataY, **kwargs):
    '''
    required inputs:
    func: the model for non-linear confidence test
    param1: first parameter - optimal value
    param2: second parameter - optimal value
    dataY: measured data

    optional inputs:
    perturb1: how much to perturb the 1st parameter (default = 0.25 or 25%)
    perturb2: how much to perturb the 2nd parameter (default = 0.25 or 25%)
    numPoints1: the number of points for first parameter (default = 25)
    numPoints2: the number of points for the second parameter (default = 25)
    args: any arguments for the model inputted as arrays
    alpha: confidence interval level (default = 0.95)
    save: boolean to save the contour plot
    show: boolean to show the contour plot

    outputs:
    a contour object with the x-values of the confidence interval to superimpose
    on a different graph
    '''

    perturb1 = kwargs.get('perturb1', 0.25)
    perturb2 = kwargs.get('perturb2', 0.25)
    numPoints1 = kwargs.get('pts1', 25)
    numPoints2 = kwargs.get('pts2', 25)
    args = kwargs.get('args', ())
    alpha = kwargs.get('alpha', 0.05)
    save = kwargs.get('save', False)
    show = kwargs.get('show', True)
    name = kwargs.get('name', 'Contour')
    title = kwargs.get('title', '95 % Confidence Region')

    param1Range = np.linspace(param1 * (1 - perturb1), param1 * (1 + perturb1), numPoints1)
    param2Range = np.linspace(param2 * (1 - perturb2), param2 * (1 + perturb2), numPoints2)

    param1_grid, param2_grid = np.meshgrid(param1Range, param2Range)
    numberParams = 2
    numberMeas = len(dataY)

    (xLength, yLength) = param1_grid.shape
    modelValues = np.zeros((xLength, yLength, numberMeas))


    for k in range(xLength):
        for l in range(yLength):
            modelValues[k, l, :] = func(param1_grid[k][l], param2_grid[k][l], *args)
    sumSqErrors = np.zeros([xLength, yLength])

    for i in range(numberMeas):
        sumSqErrors += (modelValues[:, :, i] - dataY[i]) ** 2

    bestSumSqErrors = np.min(np.min(sumSqErrors))
    fValueSumSqErrors = (sumSqErrors - bestSumSqErrors) / bestSumSqErrors

    fStatistic = scipy.stats.f.isf(alpha, numberParams, (numberMeas - numberParams))
    fLimContour = fStatistic * numberParams / (numberMeas - numberParams)
    objLimContour = fLimContour * bestSumSqErrors + bestSumSqErrors

    print ('f-test limit for SSE fractional deviation: ' + str(fLimContour))
    plt.figure()
    CS = plt.contour(param1_grid, param2_grid, sumSqErrors)
    plt.clabel(CS, inline=1, fontsize=8)
    plt.title(title)
    plt.xlabel('E_a_1')
    plt.ylabel('E_a_2')
    # solid line to show confidence region
    CS = plt.contour(param1_grid,param2_grid,sumSqErrors,[objLimContour],colors='b',linewidths=[4.0])
    plt.clabel(CS, inline=1, fontsize=15)
    CS.collections[0].set_label('%d %% Confidence Region' %((1-alpha) * 100))
    plt.legend()
    if save:
        name += ".png"
        plt.savefig(name)
    if show:
        plt.show()


    return CS.collections[0].get_paths()[0]
