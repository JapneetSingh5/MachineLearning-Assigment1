from math import sqrt,exp
from pickletools import markobject
import numpy as np
import sys
import time
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d


def L(X, y, theta):
    """log-likelihood of logistic regression."""
    return -1 * (np.sum(y * np.log(h(X, theta)) + (1 - y) * np.log(1 - h(X, theta))))

def newtons(X, y):
    theta = np.zeros(3)
    iters = 0
    converged = False

    Ln = L(X, y, theta)
    print("Initial Error: ", Ln)

    while not converged:
        theta += np.linalg.pinv(hessian(X, theta)) @ gradient(X, y, theta)

        Lp = Ln
        Ln = L(X, y, theta)

        if Lp - Ln < 10**-12:
            converged = True

        iters += 1

    print("Final Error: ", Ln)
    print("Number of iterations: ", iters)
    print("Parameters: ", theta)

    return theta

def build_h_theta_x(X, theta):
    return (1/(1 + np.exp(-np.dot(X,theta))))

def build_gradient(X, Y, theta):
    return np.dot(X.T, (build_h_theta_x(X,theta)-Y))

def build_hessian(X, theta):
    h_theta_x = build_h_theta_x(X, theta)
    diagonal = np.identity(X.shape[0]) * np.dot(h_theta_x.T,(1-h_theta_x))
    # print('diagonal', diagonal, diagonal.shape)
    return np.dot(X.T, np.dot(diagonal, X))

def theta_from_newtons(X, Y, theta):
    count = 0
    curr_theta =  theta
    temp_theta = theta
    print("Intial theta was: ", theta)
    while(np.linalg.norm(curr_theta-temp_theta)<1e-10 or count==0):
        count = count + 1
        gradient = build_gradient(X, Y, curr_theta)
        hessian = build_hessian(X, curr_theta)
        temp_theta = curr_theta
        curr_theta = curr_theta - np.dot(np.linalg.inv(hessian),gradient)
        print("Theta at iter {} is  {}", count, curr_theta)
    print("Final theta was: {}", curr_theta)
    return curr_theta

        

def main():
    # process command line arguments 
    train_x_file_extension = '/logisticX.csv'
    train_y_file_extension = '/logisticY.csv'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided")
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    df = pd.DataFrame()
    # Add X0 for intercept term
    df = pd.read_csv(train_data_x, header=None)
    df[2] = 1
    # normalize data to zero mean, unit variance
    # Normalize X1
    print("np.mean", np.mean(df.to_numpy()), "np.std", np.std(df.to_numpy()))
    print(df[0].mean(), df[0].std(),np.mean(df[0].to_numpy()), np.std(df[0].to_numpy()))
    x1_mean = np.mean(df[0].to_numpy())
    x1_std = np.std(df[0].to_numpy())
    df[0] = (df[0] - x1_mean)/x1_std;
    print(df[0].mean(), df[0].std(),np.mean(df[0].to_numpy()), np.std(df[0].to_numpy()))
    # Normalize X2
    x2_mean = np.mean(df[1].to_numpy())
    x2_std = np.std(df[1].to_numpy())
    df[1] = (df[1] - x2_mean)/x2_std;
    print(df[1].mean(), df[1].std())
    df = df[[2,1,0]]
    X = df.to_numpy()
    print("X", X)
    df[3] = pd.read_csv(train_data_y, header=None)
    print(df)
    
    # intial value of theta set to [0,0,0]
    theta = np.zeros((3, 1))
    # seperate points for Y=0, Y=1 classes
    x0_for_y0 = df.loc[df[3]==0].reset_index(drop=True)[0]
    x1_for_y0 = df.loc[df[3]==0].reset_index(drop=True)[1]
    x0_for_y1 = df.loc[df[3]==1].reset_index(drop=True)[0]
    x1_for_y1 = df.loc[df[3]==1].reset_index(drop=True)[1]
    Y = df[3].to_numpy().reshape(-1,1)
    print(Y, Y.shape)

    h_theta_x = build_h_theta_x(X, theta)
    print("h_theta_x", h_theta_x)
    gradient = build_gradient(X, Y, theta)
    print('X.t', X.T, (h_theta_x-Y))
    print(gradient)
    
    diagonal = np.identity(X.shape[0]) * np.dot(h_theta_x.T,(1-h_theta_x))
    # print('diagonal', diagonal, diagonal.shape)
    hessian = np.dot(X.T, np.dot(diagonal, X))
    print(hessian)


    # # Newton's Update Equation for theta (converges in one step for quadratic and linear)
    theta_new = theta - np.dot(np.linalg.inv(hessian),gradient)
    print("The final Theta from Newton's Method = \n", theta_new)
    # # # print(df)

    # # x_val = np.array([np.min(X[:, 1] ), np.max(X[:, 1] )]).reshape(1,-1)
    # # print(x_val)
    # # y_val = np.dot((-1./theta_new[2:3]),np.dot(theta_new[1:2], x_val)) - theta_new[0:1]
    # # print(y_val)
    plt.figure(1, figsize=(16, 9))
    plt.scatter(x0_for_y0, x1_for_y0, color='red', marker='_', label='Y=0')
    plt.scatter(x0_for_y1, x1_for_y1, color='green', marker='+', label='Y=1')
    ax = plt.gca()
    x_val = ax.get_xlim()
    # y_val = -1*(theta_new[0] + theta_new[1] * x_val)/theta_new[2]
    # # plt.plot(x_val, y_val, color='g')

    plt.xlabel('Feature X1')
    plt.ylabel('Feature X2')
    plt.legend(loc='upper left')
    
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    plt.show()
    
    


if __name__ == "__main__":
    main()


