from math import sqrt
from pickletools import markobject
import numpy as np
import sys
import time
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d

def plot_mesh(x, y):
    n = x.size[0] 

def average_error(x, y, a, b):
    m = x.size 
    aggregate_error = 0
    for i in range(0, m):
        aggregate_error += (y[i] - a*x[i] - b)**2;
    return aggregate_error/(2*m);
    # todo - divide cost by m or not? removing m gives better result

def lms_update_a(x, y, a, b):
    m = x.size 
    aggregate_error = 0
    for i in range(0, m):
        aggregate_error += (y[i] - a*x[i] - b)*x[i];
    return aggregate_error;

def lms_update_b(x, y, a, b):
    m = x.size 
    aggregate_error = 0
    for i in range(0, m):
        aggregate_error += (y[i] - a*x[i] - b)*1;
    return aggregate_error;

def batch_gradient_descent(x, y, learning_rate, stopping_criteria, max_iterations):
    # y is density of wine
    # x is acidity of wine
    # y = ax + b, theta = (a,b)
    ab_list = [[0,0,average_error(x, y, 0, 0)]];
    a = 0
    b = 0
    iteration_count = 0
    while True:
        iteration_count += 1
        a_new = a + learning_rate*(lms_update_a(x,y,a,b))
        b_new = b + learning_rate*(lms_update_b(x,y,a,b))
        if(average_error(x,y,a_new,b_new)<stopping_criteria or iteration_count>max_iterations):
            ab_list.append([a_new,b_new,average_error(x,y,a_new,b_new)])
            return (a_new,b_new,ab_list)
        a = a_new
        b = b_new
        ab_list.append([a,b,average_error(x,y,a,b)])
        # print(a_new, b_new)

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
    df = pd.read_csv(train_data_x, header=None)
    # normalize data to zero mean, unit variance
    x_mean = df[0].mean()
    x_std = df[0].std()
    df[0] = df.transform(lambda x: ((x-x_mean)/x_std))
    df[1] = pd.read_csv(train_data_y, header=None)
    # Q1 Part (A) - performing batch gradient descent on normalized input data
    (a,b,ab_list) = batch_gradient_descent(df[0], df[1], 0.001, 1e-10, 1000)
    print(a,b)
    print(average_error(df[0],df[1],a,b))
    df[2] = df[0]*a + b

    # Q1 Part (B) - plot of data and hypothesis function
    plt.figure(1)
    plt.scatter(df[0], df[1])
    plt.plot(df[0], df[2], color="red")
    plt.savefig("DataAndHypothesis.png")


    # Q1 Part (C) - 3D Mesh showing error function and error value at each iteration
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')

    def zfn(a, b):
        return average_error(df[0], df[1], a, b)

    theta_x = np.linspace(-1, 1, 20)
    theta_y = np.linspace(0, 2, 20)
    X, Y = np.meshgrid(theta_x, theta_y)
    Z = np.zeros(X.shape)
    for i in range(0,Z.shape[0]):
        for j in range(0,Z.shape[1]):
            Z[j][i] = average_error(df[0], df[1], X[i][j], Y[i][j])
    axes = fig.gca(projection ='3d')
    axes.plot_surface(X, Y, Z, rstride=1, cmap='summer', alpha=0.8, zorder=0)
    for i in range(0, len(ab_list)):
        if(i>100):
            break
        plt.plot(ab_list[i][0], ab_list[i][1], ab_list[i][2], c='red', marker='o',zorder=2)
        plt.savefig("GradientDescentWithCost.png")
        plt.pause(0.2)
    

    # Visualisation of how cost decays
    plt.figure(3)
    plt.scatter(range(1, len(ab_list)+1), [ele[2] for ele in ab_list], c='red', marker='o', s=10, zorder=10)
    plt.savefig("ErrorIterations.png")

    # Q1 Part (D) 
    plt.figure(4)
    plt.contour(X, Y, Z, 20, cmap='cool')
    for i in range(0, len(ab_list)):
        if(i>100):
            break
        plt.plot(ab_list[i][0], ab_list[i][1], c='red', marker='o',zorder=2)
        plt.savefig("Contour1.png")
        plt.pause(0.2)

    # Q1 Part (E)
    plt.figure(5)
    plt.contour(X, Y, Z, 20, cmap='cool')
    (_,_,ab_list) = batch_gradient_descent(df[0], df[1], 0.025, 1e-10, 1000)
    for i in range(0, len(ab_list)):
        if(i>100):
            break
        plt.plot(ab_list[i][0], ab_list[i][1], c='red', marker='o',zorder=2)
        plt.savefig("Contour2.png")
        plt.pause(0.2)
    
    plt.figure(6)
    plt.contour(X, Y, Z, 20, cmap='cool')
    (_,_,ab_list) = batch_gradient_descent(df[0], df[1], 0.1, 1e-10, 1000)
    for i in range(0, len(ab_list)):
        if(i>100):
            break
        plt.plot(ab_list[i][0], ab_list[i][1], c='red', marker='o',zorder=2)
        plt.savefig("Contour3.png")
        plt.pause(0.2)
    


if __name__ == "__main__":
    main()


