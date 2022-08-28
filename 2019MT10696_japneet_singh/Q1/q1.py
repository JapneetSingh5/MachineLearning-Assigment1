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

def batch_gradient_descent(x, y):
    # y is density of wine
    # x is acidity of wine
    # y = ax + b, theta = (a,b)
    ab_list = [[0,0,average_error(x, y, 0, 0)]];
    a = 0
    b = 0
    learning_rate = 0.001
    stopping_criteria = 1e-10
    while True:
        a_new = a + learning_rate*(lms_update_a(x,y,a,b))
        b_new = b + learning_rate*(lms_update_b(x,y,a,b))
        if(sqrt((a_new - a)**2 + (b_new - b)**2)<stopping_criteria):
            ab_list.append([a_new,b_new,average_error(x,y,a_new,b_new)])
            return (a_new,b_new,ab_list)
        a = a_new
        b = b_new
        ab_list.append([a,b,average_error(x,y,a,b)])
        # print(a_new, b_new)

def main():
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
    # print(df[0].mean(), df[0].std())
    df[1] = pd.read_csv(train_data_y, header=None)
    (a,b,ab_list) = batch_gradient_descent(df[0], df[1])
    print(a,b)
    print(average_error(df[0],df[1],a,b))
    plt.scatter(df[0], df[1])
    plt.scatter(df[0], df[0].transform(lambda x: a*x + b))
    # plt.show()
    # time.sleep(10)
    # plt.close()
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    def zfn(a, b):
        return average_error(df[0], df[1], a, b)
    theta_x = np.linspace(-0.002, 0.002, 50)
    theta_y = np.linspace(0.7, 1, 50)
    X, Y = np.meshgrid(theta_x, theta_y)
    Z = zfn(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, color='green')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                    cmap='winter', edgecolor='none')
    ax.set_title('surface');
    ax.scatter([ele[0] for ele in ab_list], [ele[1] for ele in ab_list], [ele[2] for ele in ab_list], c='red', marker='o', s=10)
    plt.show()
    # time.sleep(10)
    # plt.close()
    fig2, ax2 = plt.subplots(1, 1)
    cp = ax2.contourf(X, Y, Z)
    # ax2.scatter([ele[0] for ele in ab_list], [ele[1] for ele in ab_list], c='red', marker='o', s=10)
    ax2.set_title('Filled Contour Plot')
    ax2.set_xlabel('a')
    ax2.set_ylabel('b')
    fig2.colorbar(cp)
    plt.show()

if __name__ == "__main__":
    main()


