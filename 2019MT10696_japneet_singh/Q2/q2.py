from math import sqrt,fabs
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

def average_error(X, data_y, theta):
    return float(np.dot((np.dot(X, theta)-data_y).T, np.dot(X, theta)-data_y))/(2.0*len(X))

def build_gradient(X, data_y, theta):
    return np.dot(X.T, (np.dot(X, theta)-data_y))/len(X)

def stochastic_gradient_descent(X_Y, learning_rate, batch_size, maximum_epoch_limit, cut_off, epoch_size):
    print("Starting stochastic gradient descent...")
    print("Learning rate: " + str(learning_rate))
    print("Batch size: " + str(batch_size))
    print("Epoch size: " + str(epoch_size))
    print("Maximum number of epochs allowed: " + str(maximum_epoch_limit))
    sample_count = X_Y.shape[0]
    sample_dimension_X = X_Y.shape[1]-1
    theta = np.matrix([[0]] * (sample_dimension_X))
    epoch_count = 0
    iter_count = 0
    batch_count = int(sample_count/batch_size)
    curr_avg = 0.0
    cost_history = []
    theta_history = []
    while(True):
        epoch_count = epoch_count + 1
        if epoch_count > maximum_epoch_limit:
            print("Epoch limit exceeded, stochastic gradient descent hasn't converged")
            print("Stopping algo")
            break
        print("Currently at epoch number: " + str(epoch_count))
        np.random.shuffle(X_Y)
        X = X_Y[:, 0:3]
        Y = X_Y[:, 3:4]
        for batch_number in range(0, batch_count):
            first_index = batch_number*batch_size
            end_index = first_index + batch_size
            x_sample = X[first_index:end_index]
            y_sample = Y[first_index:end_index]
            curr_avg = curr_avg + average_error(x_sample, y_sample, theta)
            theta = theta - build_gradient(x_sample, y_sample, theta)*learning_rate
            iter_count = iter_count + 1
            if iter_count == epoch_size:
                curr_avg = curr_avg/epoch_size
                cost_history.append(curr_avg)
                theta_history.append(np.array(theta))
                if len(cost_history) > 1:
                    if fabs(cost_history[-1]-cost_history[-2]) < cut_off:
                        print("Converged agt epoch number:"+str(epoch_count))
                        return [theta, cost_history, theta_history]
                curr_avg = 0.0
                iter_count = 0
    return [theta, cost_history, theta_history]

def main():
    args_length = len(sys.argv);
    if(args_length<2):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_test_data = sys.argv[1]
    mu1 = 3
    sigma1 = 2
    mu2 = -1
    sigma2 = 2
    mu_noise = 0
    sigma_noise = sqrt(2)
    x1 = np.random.normal(mu1, sigma1, 1000000);
    x2 = np.random.normal(mu2, sigma2, 1000000);
    noise = np.random.normal(mu_noise, sigma_noise, 1000000);
    theta_sample = [3, 1, 2]
    # Q2 Part (A) Generating Sample Data
    samples = pd.DataFrame();
    samples[0] = x1
    samples[1] = x2
    samples[2] = 1
    samples = samples[[2, 0, 1]]
    X = samples.to_numpy()
    print(X)
    # building Y values from initial theta set to (3,1,2)
    samples[3] = theta_sample[0]*1 + theta_sample[1]*x1 + theta_sample[2]*x2 + noise;
    X_Y = samples.to_numpy()
    print(X_Y)
    # Q2 Part (B) Stochastic Gradient Descent
    result_list_b1 = stochastic_gradient_descent(X_Y, 0.001, 1, 10000, 1e-4, 1000)
    result_list_b2 = stochastic_gradient_descent(X_Y, 0.001, 100, 10000, 1e-4, 1000)
    result_list_b3 = stochastic_gradient_descent(X_Y, 0.001, 10000, 10000, 1e-4, 1000)
    result_list_b4 = stochastic_gradient_descent(X_Y, 0.001, 1000000, 30000, 1e-4, 100)

    plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.set_title("Movement of Theta, Batch Size 1")
    ax.set_xlabel("Theta0")
    ax.set_ylabel("Theta1")
    ax.set_zlabel("Theta2")
    theta_history1 = np.array(result_list_b1[2])
    reshape_size1 = theta_history1.shape[0]
    theta_history1 = theta_history1.reshape(reshape_size1, 3)
    theta1_0 = np.array(theta_history1[:, 0:1]).reshape(reshape_size1)
    theta1_1 = np.array(theta_history1[:, 1:2]).reshape(reshape_size1)
    theta1_2 = np.array(theta_history1[:, 2:3]).reshape(reshape_size1)
    ax.set_xlim([np.min(theta1_0), np.max(theta1_0)])
    ax.set_ylim([np.min(theta1_1), np.max(theta1_1)])
    ax.set_zlim([np.min(theta1_2), np.max(theta1_2)])
    ax.scatter(theta1_0, theta1_1, theta1_2)
    plt.savefig("Theta3DGraphBatchSize1.png")

    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.set_title("Movement of Theta, Batch Size 100")
    ax.set_xlabel("Theta0")
    ax.set_ylabel("Theta1")
    ax.set_zlabel("Theta2")
    theta_history2 = np.array(result_list_b2[2])
    reshape_size2 = theta_history2.shape[0]
    theta_history2 = theta_history2.reshape(reshape_size2, 3)
    theta2_0 = np.array(theta_history2[:, 0:1]).reshape(reshape_size2)
    theta2_1 = np.array(theta_history2[:, 1:2]).reshape(reshape_size2)
    theta2_2 = np.array(theta_history2[:, 2:3]).reshape(reshape_size2)
    ax.set_xlim([np.min(theta2_0), np.max(theta2_0)])
    ax.set_ylim([np.min(theta2_1), np.max(theta2_1)])
    ax.set_zlim([np.min(theta2_2), np.max(theta2_2)])
    ax.scatter(theta2_0, theta2_1, theta2_2)
    plt.savefig("Theta3DGraphBatchSize100.png")

    plt.figure(3)
    ax = plt.axes(projection='3d')
    ax.set_title("Movement of Theta, Batch Size 10000")
    ax.set_xlabel("Theta0")
    ax.set_ylabel("Theta1")
    ax.set_zlabel("Theta2")
    theta_history3 = np.array(result_list_b3[2])
    reshape_size3 = theta_history3.shape[0]
    theta_history3 = theta_history3.reshape(reshape_size3, 3)
    theta3_0 = np.array(theta_history3[:, 0:1]).reshape(reshape_size3)
    theta3_1 = np.array(theta_history3[:, 1:2]).reshape(reshape_size3)
    theta3_2 = np.array(theta_history3[:, 2:3]).reshape(reshape_size3)
    ax.set_xlim([np.min(theta3_0), np.max(theta3_0)])
    ax.set_ylim([np.min(theta3_1), np.max(theta3_1)])
    ax.set_zlim([np.min(theta3_2), np.max(theta3_2)])
    ax.scatter(theta3_0, theta3_1, theta3_2)
    plt.savefig("Theta3DGraphBatchSize10000.png")

    plt.figure(4)
    ax = plt.axes(projection='3d')
    ax.set_title("Movement of Theta, Batch Size 1000000")
    ax.set_xlabel("Theta0")
    ax.set_ylabel("Theta1")
    ax.set_zlabel("Theta2")
    theta_history4 = np.array(result_list_b4[2])
    reshape_size4 = theta_history4.shape[0]
    theta_history4 = theta_history4.reshape(reshape_size4, 3)
    theta4_0 = np.array(theta_history4[:, 0:1]).reshape(reshape_size4)
    theta4_1 = np.array(theta_history4[:, 1:2]).reshape(reshape_size4)
    theta4_2 = np.array(theta_history4[:, 2:3]).reshape(reshape_size4)
    ax.set_xlim([np.min(theta4_0), np.max(theta4_0)])
    ax.set_ylim([np.min(theta4_1), np.max(theta4_1)])
    ax.set_zlim([np.min(theta4_2), np.max(theta4_2)])
    ax.scatter(theta4_0, theta4_1, theta4_2)
    plt.savefig("Theta3DGraphBatchSize1000000.png")
        
if __name__ == "__main__":
    main()
