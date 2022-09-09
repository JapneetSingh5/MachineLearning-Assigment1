from math import sqrt,fabs
from pickletools import markobject
import numpy as np
import sys
import time
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

def cost(data_x, data_y, theta):
    cost_val = np.subtract(np.dot(data_x, theta), data_y)
    return float((np.dot(cost_val.T, cost_val)))/(2.0*len(data_x))

def grad(data_x, data_y, theta):
    grad_val = np.subtract(np.dot(data_x, theta), data_y) # found bug here # removed y reshape for now  np.reshape(data_y, (n, 1))
    grad_val = np.dot(data_x.T, grad_val)
    return grad_val/len(data_x)

def batch_gradient_descent(data_x, data_y, learning_rate, cut_off, max_iterations):
    theta = np.zeros((2, 1), dtype=float)
    store =[]
    curr_cost = 0.0
    temp_cost = 0.0
    i = 0
    while(fabs(curr_cost - temp_cost) > cut_off or i==0):
        if i > max_iterations:
            print("Not converging\n")
            sys.exit(1)
        store.append([float(theta[0]), float(theta[1]), float(curr_cost)])
        temp_cost = curr_cost
        curr_cost = cost(data_x, data_y, theta)
        gradient = grad(data_x, data_y, theta)
        i += 1
        theta -= (learning_rate)*(gradient)

    print("learning_rate for this run: {}".format(learning_rate))
    print("Final cost on convergence: {}".format(curr_cost))
    print("Number of iterations: {}".format(i))
    return [theta,store]

def main():
    # process command line arguments 
    train_x_file_extension = '/linearX.csv'
    train_y_file_extension = '/linearY.csv'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided")
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    df = pd.DataFrame()
    df = pd.read_csv(train_data_x, header=None)
    # normalize data to zero mean, unit variance
    x_mean = df[0].mean()
    x_std = df[0].std()
    df[0] = df[0].transform(lambda x: ((x-x_mean)/x_std))
    df[1] = 1
    df = df[[1,0]]
    X = df.to_numpy()
    df[2] = pd.read_csv(train_data_y, header=None)
    Y = df[2].to_numpy()
    Y = Y.reshape(-1, 1)
    print(X, Y)

    # Q1 Part (A) - performing batch gradient descent on normalized input data
    (theta1,ab_list) = batch_gradient_descent(X, Y, 0.01, 1e-10, 10000)
    Y1 =  np.matmul(X, theta1)

    # Q1 Part (B) - plot of data and hypothesis function obtained in Part A
    plt.figure(1)
    plt.scatter(df[0].to_numpy(), Y.ravel())
    plt.plot(df[0].to_numpy(), Y1.ravel(), color="red")
    plt.savefig("DataAndHypothesis.png")


    # Q1 Part (C) - 3D Mesh showing error function and error value at each iteration
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')


    theta_x = np.linspace(-1, 1, 20)
    theta_y = np.linspace(0, 2, 20)
    X3d, Y3d = np.meshgrid(theta_x, theta_y)
    Z = np.zeros(X3d.shape)
    for i in range(0,Z.shape[0]):
        for j in range(0,Z.shape[1]):
            theta_temp= np.array([[Y3d[i][j], X3d[i][j]]])
            theta_temp = np.transpose(theta_temp)
            # print(theta_temp)
            # print( np.array([[X[i][j]],[ Y[i][j]]]) )
            Z[j][i] = cost(X, Y, theta_temp)
    axes = fig.gca()
    axes.plot_surface(X3d, Y3d, Z, rstride=1, cmap='summer', alpha=0.8, zorder=0)
    for i in range(0, len(ab_list)):
        if(i>1000):
            break
        plt.plot(ab_list[i][1], ab_list[i][0], ab_list[i][2], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("GradientDescentWithCost.png")
    

    # Visualisation of how cost decays
    plt.figure(3)
    plt.scatter(range(1, len(ab_list)+1), [ele[2] for ele in ab_list], c='red', marker='o', s=10, zorder=10)
    plt.savefig("ErrorIterations.png")

    # Q1 Part (D) 
    plt.figure(4)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    for i in range(0, len(ab_list)):
        if(i>1000):
            break
        plt.plot(ab_list[i][1], ab_list[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour1.png")

    # Q1 Part (E)
    plt.figure(5)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    (theta1,ab_list2) = batch_gradient_descent(X, Y, 0.025, 1e-6, 10000)
    for i in range(0, len(ab_list2)):
        if(i>1000):
            break
        plt.plot(ab_list2[i][1], ab_list2[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour2.png")
    
    plt.figure(6)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    (theta1,ab_list3) = batch_gradient_descent(X, Y, 0.1, 1e-4, 10000)
    for i in range(0, len(ab_list3)):
        if(i>1000):
            break
        plt.plot(ab_list3[i][1], ab_list3[i][0], c='red', marker='o',zorder=2)
        plt.savefig("Contour3.png")
        plt.pause(0.2)
    

if __name__ == "__main__":
    main()


