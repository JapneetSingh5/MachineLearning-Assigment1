from math import sqrt,exp
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
    # print('diagonal', diagonal, diagonal.shape)
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

def main():
    # Step 1. Process command line arguments and build input data structures
    train_x_file_extension = '/logisticX.csv'
    train_y_file_extension = '/logisticY.csv'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided")
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    # Step 2. Prepare input data
    # 2(i) Add X0 for intercept term
    df = pd.read_csv(train_data_x, header=None)
    df[2] = 1
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
    df = df[[2,0,1]]
    X = df.to_numpy()
    # print("X", X)
    # 2(vi) build y values NumPy array
    df[3] = pd.read_csv(train_data_y, header=None)
    Y = df[3].to_numpy().reshape(-1,1)
    # print(Y, Y.shape)

    # Step 3. Plot training data on a scatter plot
    x0_for_y0 = df.loc[df[3]==0].reset_index(drop=True)[0]
    x1_for_y0 = df.loc[df[3]==0].reset_index(drop=True)[1]
    x0_for_y1 = df.loc[df[3]==1].reset_index(drop=True)[0]
    x1_for_y1 = df.loc[df[3]==1].reset_index(drop=True)[1]
    plt.figure(1, figsize=(16, 9))
    plt.scatter(x0_for_y0, x1_for_y0, color='red', marker='_', label='Y=0')
    plt.scatter(x0_for_y1, x1_for_y1, color='green', marker='+', label='Y=1')
    plt.xlabel('Feature X1')
    plt.ylabel('Feature X2')
    plt.legend(loc='upper right')
    plt.title('Training Data for Q3')
    plt.savefig('TrainingData.png')
    
    # Step 4. Use Newton's Method to get correct parameters
    # intial value of theta set to [0,0,0]
    theta = np.zeros((3, 1))
    theta_optimal = theta_from_newtons(X, Y, theta)
    
    # Step 5. Plot decision boundary
    ax = plt.gca()
    x_val = ax.get_xlim()
    y_val = -1*(theta_optimal[0] + theta_optimal[1] * x_val)/theta_optimal[2]
    plt.plot(x_val, y_val, color='b')
    ax.set_xlim([np.min(X[:, 1]), np.max(X[:, 1]) ])
    ax.set_ylim([np.min(X[:, 2]), np.max(X[:, 2]) ])
    plt.title('Decision Boundary with Training Data')
    plt.savefig('DecisionBoundary.png')
    # plt.show()

if __name__ == "__main__":
    main()


