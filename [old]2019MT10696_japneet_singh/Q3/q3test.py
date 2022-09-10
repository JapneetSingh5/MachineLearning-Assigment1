import math
import sys
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import time
# from random import shuffle, random
# plt.rcParams.update({'font.size': 12})


# learning_rate = 0.01
cut_off = pow(10, -13)

plot = True
# plot = False

def normalise(data_x):
    # data_x = (data_x - avg) / standard dev
    # there is a difference between normalisation and standardization
        # in class standardization was discussed as normalisation, so I am doing that
        # anyways, standardization is better than normalisation
    avg_x = np.mean(data_x, axis=0, dtype=np.float64)
    print("avg_x",avg_x)
    std_x = np.std(data_x, axis=0, dtype=np.float64)
    print("std",std_x)
    data_x = (data_x - avg_x)/(std_x) # this works only for one feature data
    print(data_x)
    return data_x


def plot(ones, zero, data_x, data_y, flag, theta):
    data_ones = np.array([data_x[_, 1:3] for _ in ones])
    data_zero = np.array([data_x[_, 1:3] for _ in zero])
    plt.xlabel("x1")
    plt.ylabel("x2")
    axes = plt.gca()
    axes.set_xlim([np.min(data_x[:, 1]), np.max(data_x[:, 1]) ])
    axes.set_ylim([np.min(data_x[:, 2]), np.max(data_x[:, 2]) ])
    # if flag == "initial":
    plt.plot(data_ones[:, 0], data_ones[:, 1], 'yo', marker=".", label="1s")
    plt.plot(data_zero[:, 0], data_zero[:, 1], 'ro', marker="+", label="0s")
    plt.legend()
    plt.title("Training Data")
    plt.savefig("training_data.png")

    if flag == "boundary":
        boundary = np.arange(np.min(data_x[:, 1]), np.max(data_x[:, 1]), 0.01)
        y_boundary = -(theta.item(1)*boundary + theta.item(0))/(theta.item(2))
        plt.plot(boundary, y_boundary, 'b-')
        plt.title("Training Data with learned boundary")
        plt.savefig("learned_boundary.png")
    plt.show(block=False)
    input("Press enter to continue.\n")
    plt.close()


def sigmoid(data):
    return 1/(1 + np.exp(-data))

# def cost(data_x, data_y, theta):
#     n = np.shape(data_x)[0]
#     cost_val = np.subtract(sigmoid(np.dot(data_x, theta)), data_y)
#     return float((np.dot(cost_val.T, cost_val)))/(2.0*len(data_x))

def grad(data_x, data_y, theta):
    grad_val = np.subtract(data_y, sigmoid(np.dot(data_x, theta))) # found bug here # removed y reshape for now  np.reshape(data_y, (n, 1))
    grad_val = np.dot(data_x.T, grad_val)
    return grad_val


# referred stackoverflow for hessian
def hessian(data_x, theta): # re check this
    n = np.shape(theta)[0]
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            sum = 0
            for k in range(data_x.shape[0]):
                hyp = sigmoid(np.dot(data_x[k, :], theta))
                sum = sum + (data_x[k, i] * data_x[k, j] * hyp * (hyp - 1))
            H[i, j] = sum
    return H

def newton_method(data_x, data_y, theta):
    # theta := theta - H_inverse * del theta f(theta)
    prev_theta = theta
    diff = 1
    i = 0
    while(diff > cut_off):
        print(theta)
        # print(diff)
        gradv = grad(data_x, data_y, theta)
        print('grad', gradv)
        H = hessian(data_x, theta)
        print('H', H)
        prev_theta = theta
        theta = theta - np.dot(np.linalg.pinv(H), gradv)
        #   ABSOLUTE SHITTY BUG: Took more than 3 hours to find   https://stackoverflow.com/questions/9047111/vs-operators-with-numpy
        # theta -= np.dot(np.linalg.pinv(H), gradv)
        i += 1
        diff = np.max(np.abs(theta - prev_theta))

    print("Number of iterations: {}".format(i))
    return theta

if __name__ == '__main__':
    data_x = []
    data_y = []
    dir = os.path.dirname(__file__)
    x_filename = os.path.normpath('/Users/japneet/Desktop/COL774/COL774-Assignment1/data/q3/logisticX.csv')
    y_filename = os.path.normpath('/Users/japneet/Desktop/COL774/COL774-Assignment1/data/q3/logisticY.csv')
    count_x = 0
    with open(x_filename) as linearX:
        csv_reader = csv.reader(linearX, delimiter=',')
        for row in csv_reader:
            count_x += 1
            data_row = []
            data_row.append(float(row[0])) # row is a list of strings, row[0] is a string, float() converts it to float
            data_row.append(float(row[1]))
            data_x.append(data_row)

    count_y = 0
    with open(y_filename) as linearY:
        csv_reader = csv.reader(linearY, delimiter=',')
        for row in csv_reader:
            count_y += 1
            data_y.append(float(row[0]))

    try:
        if(count_x != count_y):
            raise ValueError()
    except ValueError:
        print("Error: Training data not coherent. Number of Ys do not match the number of Xs")
        sys.exit(1)

    n = np.shape(data_x)[0] # number of training examples
    m = 2 # only feature in our data

    data_x = np.reshape(data_x, (n,2))
    data_y = np.reshape(data_y, (n,1))

    orig_x = data_x
    orig_x_normalised = normalise(data_x)
    data_x = orig_x_normalised
    data_x = np.c_[np.ones((n, 1)), data_x] # add x0 = 1 column
    # if plot:
    #     plot_initial_data(orig_x, data_y, orig_x_normalised, "normal", None) # shows normalised data with original data in one plot
    # # print(data_x)

    ones = []
    zero = []
    for i in range(0, n):
        if data_y[i] == 1:
            ones.append(i)
        else:
            zero.append(i)

    theta = np.zeros((3, 1), dtype=float)
    if plot:
        plot(ones, zero, data_x, data_y, "initial", theta)
    theta = newton_method(data_x, data_y, theta)
    print("Learned theta value is: \n {}".format(theta))
    if plot:
        plot(ones, zero, data_x, data_y, "boundary", theta)