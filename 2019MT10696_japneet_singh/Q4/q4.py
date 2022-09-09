from math import sqrt,exp,log
from numbers import Number
from pickletools import markobject
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt 


def build_h_theta_x(X, theta):
    return (1/(1 + np.exp(-np.dot(X,theta))))

def build_gradient(X, Y, theta):
    return np.dot(X.T, (build_h_theta_x(X,theta)-Y))

def build_hessian(X, theta):
    h_theta_x = build_h_theta_x(X, theta)
    diagonal = np.identity(X.shape[0]) * np.dot(h_theta_x.T,(1-h_theta_x))
    return np.dot(X.T, np.dot(diagonal, X))

def theta_from_newtons(X, Y, theta):
    count = 0
    curr_theta =  theta
    temp_theta = theta
    print("Intial theta was: ", theta)
    while( ( np.max(np.abs((curr_theta-temp_theta)))>1e-4) or count==0):
        count = count + 1
        gradient = build_gradient(X, Y, curr_theta)
        print('gradient', gradient)
        hessian = build_hessian(X, curr_theta)
        print('hessian', hessian)
        temp_theta = curr_theta
        curr_theta = curr_theta - np.dot(np.linalg.inv(hessian),gradient)
        print("Theta at iter", count," is ", curr_theta)
    print("Final theta was", curr_theta)
    return curr_theta

def build_sigma_zero(X, Y_bin, u_zero, num_zero):
    s0 = np.zeros((2, 2))
    for i in range(len(X)):
        if Y_bin[i] == 0:
            diff = X[i]-u_zero
            s0 += np.outer(diff, diff)
    return s0/num_zero

def build_sigma_one(X, Y_bin, u_one, num_one):
    s1 = np.zeros((2, 2))
    for i in range(len(X)):
        if Y_bin[i] == 1:
            diff = X[i]-u_one
            s1 += np.outer(diff, diff)
    return s1/num_one

def main():
    # Step 1. Process command line arguments and build input data structures
    train_x_file_extension = '/q4x.dat'
    train_y_file_extension = '/q4y.dat'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided")
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    # Step 2. Prepare input data
    # 2(i) Add X0 for intercept term
    df = pd.read_csv(train_data_x, header=None,sep='  ',engine='python')
    # df[2] = 1
    # 2(ii) Normalize data to zero mean, unit variance
    # 2(iii) Normalize X1
    # print("np.mean", np.mean(df.to_numpy()), "np.std", np.std(df.to_numpy()))
    # print(df[0].mean(), df[0].std(),np.mean(df[0].to_numpy()), np.std(df[0].to_numpy()))
    # NumPy giving more precise mean values than Pandas, hence used its mean function
    x1_mean = np.mean(df[0].to_numpy())
    x1_std = np.std(df[0].to_numpy())
    df[0] = (df[0] - x1_mean)/x1_std;
    print(df[0].mean(), df[0].std(),np.mean(df[0].to_numpy()), np.std(df[0].to_numpy()))
    # 2(iv) Normalize X2
    x2_mean = np.mean(df[1].to_numpy())
    x2_std = np.std(df[1].to_numpy())
    df[1] = (df[1] - x2_mean)/x2_std;
    print(df[1].mean(), df[1].std())
    # 2(v) build NumPy array for further usage
    X = df.to_numpy()
    # 2(vi) build y values NumPy array
    df[3] = pd.read_csv(train_data_y, header=None)
    Y = df[3].to_numpy().reshape(-1,1)
    Y_bin = np.array((Y=='Canada'), dtype=np.float)
    print(Y_bin)
    # print(Y, Y.shape)

    # Step 3. Plot training data on a scatter plot
    x0_for_y0 = df.loc[df[3]=='Alaska'].reset_index(drop=True)[0]
    x1_for_y0 = df.loc[df[3]=='Alaska'].reset_index(drop=True)[1]
    x0_for_y1 = df.loc[df[3]=='Canada'].reset_index(drop=True)[0]
    x1_for_y1 = df.loc[df[3]=='Canada'].reset_index(drop=True)[1]
    num_zero = x0_for_y0.shape[0]
    num_one = x0_for_y1.shape[0]
    phi = num_one/(num_zero +  num_one)
    u_zero = (np.dot((1-Y_bin).T, X))/num_zero
    u_one = (np.dot((Y_bin).T, X))/num_one
    print(u_zero, u_one, phi)
    sigma_zero = build_sigma_zero(X, Y_bin, u_zero, num_zero)
    sigma_one = build_sigma_one(X, Y_bin, u_one, num_one)
    sigma_common = (sigma_one*num_one + sigma_zero*num_zero)/(num_one+num_zero)
    print(sigma_zero, sigma_one, sigma_common)
    print(num_zero, num_one)

    sigma_one_inv = np.linalg.inv(sigma_one)
    sigma_zero_inv = np.linalg.inv(sigma_zero)
    sigma_common_inv = np.linalg.inv(sigma_common)

    # For Linear Decision Boundary,
    # B(X.T) + C 

    plt.figure(1, figsize=(8, 5))
    plt.scatter(x0_for_y0, x1_for_y0, color='red', marker='o', label='Y=Alaska')
    plt.scatter(x0_for_y1, x1_for_y1, color='green', marker='*', label='Y=Canada')
    plt.xlabel('Feature X1')
    plt.ylabel('Feature X2')
    plt.legend(loc='upper right')
    ax = plt.gca()
    ax.set_xlim([np.min(X[:, 0]), np.max(X[:, 0]) ])
    ax.set_ylim([np.min(X[:, 1]), np.max(X[:, 1]) ])
    plt.title('Training Data for Q3')
    plt.savefig('TrainingData.png')

    # Step 5. Plot decision boundary
    u_zero = u_zero.reshape(2,1)
    u_one = u_one.reshape(2,1)

    x1 = np.arange(-3, 3, 0.1)
    x2 = np.arange(-3, 3, 0.1)
    x1, x2 = np.meshgrid(x1, x2)
    z = np.zeros(x1.shape)

    C = log(phi/(1-phi)) + (1/2)*((np.dot(np.matmul(u_one.T, sigma_common_inv), u_one)) - (np.dot(np.matmul(u_zero.T, sigma_common_inv), u_zero)))

    for i in range(0, x1.shape[0]):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]])
            x = x.reshape(2, 1)
            z[i][j] = C + np.dot(x.T, np.matmul(sigma_common_inv, u_one-u_zero))

    plt.contour(x1, x2, z, levels=[0], color='green')
    plt.title("Training Data with learned boundary")
    plt.savefig("LinearBoundary.png")

    D = log(phi/(1-phi)) - 0.5*log(np.linalg.det(sigma_one)/np.linalg.det(sigma_zero)) 
    for i in range(0, x1.shape[0]):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]])
            x = x.reshape(2, 1)
            z[i][j] = D + 0.5*np.dot((x-u_zero).T, np.matmul(sigma_zero_inv, (x-u_zero))) - 0.5*np.dot((x-u_one).T, np.matmul(sigma_one_inv, (x-u_one)))

    plt.contour(x1, x2, z, levels=[0], color='black')
    plt.title("Training Data with learned boundary")
    plt.savefig("QuadraticBoundary.png")


    plt.show()

if __name__ == "__main__":
    main()


