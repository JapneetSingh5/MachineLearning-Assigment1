from math import fabs
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt 

def average_error(data_x, data_y, theta):
    return float(np.dot((np.dot(data_x, theta)-data_y).T, np.dot(data_x, theta)-data_y))/(2.0*len(data_x))

def build_gradient(data_x, data_y, theta):
    return np.dot(data_x.T, (np.dot(data_x, theta)-data_y))/len(data_x)

def batch_gradient_descent(data_x, data_y, learning_rate, cut_off, max_iterations):
    print("Beginning Batch Gradient Descent")
    print("Learning Rate is: " + str(learning_rate))
    print("Initial theta set to (0,0)")
    theta = np.zeros((2, 1))
    curr_cost = 0.0
    temp_cost = 0.0
    iter_counter = 0
    history =[]
    while(fabs(curr_cost - temp_cost) > cut_off or iter_counter==0):
        if iter_counter > max_iterations:
            print("The Gradient Descent algorithm has exceeded the set maximum number of iterations\n")
            print("Stopping gradient descent, did not converge after  " + str(iter_counter) + " iterations")
            sys.exit(1)
        history.append([float(theta[0]), float(theta[1]), float(curr_cost)])
        temp_cost = curr_cost
        curr_cost = average_error(data_x, data_y, theta)
        iter_counter += 1
        theta -= build_gradient(data_x, data_y, theta)*learning_rate
    print("Number of iterations taken to converge: "+str(iter_counter))
    print("Final theta parameters obtained from the gradient descent: " +  str(theta[0]) + "," + str(theta[1]))
    print("Final error value: " + str(curr_cost))
    return [theta,history]

def main():
    # process command line arguments 
    train_x_file_extension = '/linearX.csv'
    train_y_file_extension = '/linearY.csv'
    test_x_file_extension = '/X.csv'
    test_y_file_extension = '/result_1.txt'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    test_data_x = path_test_data + test_x_file_extension;
    test_result_y = path_test_data + test_y_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    # Read training X data from csv file
    df = pd.read_csv(train_data_x, header=None)
    # normalize X data to zero mean, unit variance
    x_mean = df[0].mean()
    x_std = df[0].std()
    df[0] = df[0].transform(lambda x: ((x-x_mean)/x_std))
    # Add Intercept term to  X data
    df[1] = 1
    df = df[[1,0]]
    # create numpy array for X data from dataframe 
    X = df.to_numpy()
    df[2] = pd.read_csv(train_data_y, header=None)
    Y = df[2].to_numpy()
    Y = Y.reshape(-1, 1)
    # print(X, Y)

    # Q1 Part (A) - performing batch gradient descent on normalized input data
    (theta1,ab_list) = batch_gradient_descent(X, Y, 0.015, 1e-13, 10000)
    Y1 =  np.matmul(X, theta1)

    # Q1 Part (B) - plot of data and hypothesis function obtained in Part A
    plt.figure(1)
    plt.scatter(df[0].to_numpy(), Y.ravel())
    plt.plot(df[0].to_numpy(), Y1.ravel(), color="red")
    plt.xlabel("Feature X : Acidity")
    plt.ylabel("Value Y : Density")
    plt.title("Density v/s Acidity of wine : Linear Regression")
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
            # print(np.array([[X[i][j]],[ Y[i][j]]]))
            Z[j][i] = average_error(X, Y, theta_temp)
    axes = fig.gca()
    axes.plot_surface(X3d, Y3d, Z, rstride=1, cmap='summer', alpha=0.8, zorder=0)
    # Starting from index 1 as initial error is set to 0
    print("Making 3d Map of Gaussian Descent Error Function's Values..")
    for i in range(1, len(ab_list)):
        plt.plot(ab_list[i][1], ab_list[i][0], ab_list[i][2], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("GradientDescentWithCost.png")
    plt.close()
    print("Visualising how the cost function decays..")
    # Visualisation of how cost decays
    plt.figure(3)
    plt.scatter(range(1, len(ab_list)+1), [ele[2] for ele in ab_list], c='red', marker='o', s=1, zorder=10)
    plt.xlabel("Number of Iterations ->")
    plt.ylabel("Cost function value")
    plt.title("Cost v/s Iterations")
    plt.savefig("ErrorIterations.png")
    print("Making countor plot for learning rate = 0.015")
    # Q1 Part (D) - contour for above gradient descent
    plt.figure(4)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    for i in range(0, len(ab_list)):
        if(i>1000):
            break
        plt.plot(ab_list[i][1], ab_list[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour_StepSize_0.015.png")
    print("Making countor plot for learning rate = 0.01")
    # Q1 Part (E) - contour plots for LR = 0.01, 0.025, 0.1 respectivel
    plt.figure(5)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    (theta2,ab_list2) = batch_gradient_descent(X, Y, 0.01, 1e-13, 10000)
    for i in range(0, len(ab_list2)):
        if(i>1000):
            break
        plt.plot(ab_list2[i][1], ab_list2[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour_StepSize_0.01.png")
    print("Making countor plot for learning rate = 0.025")
    plt.figure(6)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    (theta3,ab_list3) = batch_gradient_descent(X, Y, 0.025, 1e-8, 10000)
    for i in range(0, len(ab_list3)):
        if(i>1000):
            break
        plt.plot(ab_list3[i][1], ab_list3[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour_StepSize_0.025.png")
    print("Making countor plot for learning rate = 0.01")
    plt.figure(7)
    plt.contour(X3d, Y3d, Z, 20, cmap='cool')
    (theta4,ab_list4) = batch_gradient_descent(X, Y, 0.1, 1e-6, 10000)
    for i in range(0, len(ab_list4)):
        if(i>1000):
            break
        plt.plot(ab_list4[i][1], ab_list4[i][0], c='red', marker='o',zorder=2)
        plt.pause(0.2)
    plt.savefig("Contour_StepSize_0.1.png")

    df_test = pd.read_csv(test_data_x, header=None)
    df_test[0] = df_test[0].transform(lambda x: ((x-x_mean)/x_std))
    df_test = df_test*theta1[1] + theta1[0] 
    df_test.to_csv(test_result_y, index=False, header=None)

if __name__ == "__main__":
    main()


