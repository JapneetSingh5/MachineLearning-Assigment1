from math import sqrt,fabs
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt

def cost(data_x, data_y, theta):
    cost_val = np.linalg.norm(np.subtract(data_y, data_x * theta))**2
    return float(cost_val)/(2.0*len(data_x))

def grad(data_x, data_y, theta):
    grad_val = np.subtract(data_x * theta, data_y)
    grad_val = data_x.T * grad_val
    return grad_val/len(data_x)


def stochastic(all_data, learning_rate, batch_size, max_epochs, number_samples, cut_off):
    theta = np.matrix([[float(0)]] * (all_data.shape[1]-1))
    not_converged = True
    curr_cost = -1.0
    prev_cost = -1.0
    epoch = 0
    number_batches = (np.shape(all_data)[0])/(batch_size)
    number_batches = int(number_batches)
    count = 0
    prev_avg = 0.0
    curr_avg = 0.0
    store_costs = []
    store_theta = []
    val = number_samples/batch_size
    val = min(1000, val*10)
    while(not_converged):
        np.random.shuffle(all_data)
        print("Epoch number: {}".format(epoch))
        X = all_data[:, 0:3]
        Y = all_data[:, 3:4]
        for batch in range(0, number_batches):
            l = batch*batch_size
            r = l+batch_size
            batch_x = X[l:r]
            batch_y = Y[l:r]
            new_cost = cost(batch_x, batch_y, theta)
            curr_avg = curr_avg + new_cost
            gradv = grad(batch_x, batch_y, theta)
            theta = theta - (learning_rate)*(gradv)
            count += 1
            if count == val:
                curr_avg = curr_avg/val
                store_costs.append(curr_avg)
                store_theta.append(np.array(theta))
                curr_avg = 0.0
                count = 0
                if len(store_costs) > 1:
                    if fabs(store_costs[-1]-store_costs[-2]) < cut_off:
                        print("Number of epochs taken: {}".format(epoch))
                        return [theta, store_costs, store_theta]
        epoch += 1
        if epoch > max_epochs:
            print("Taking too much time. Look at plot.\n")
            break
    return theta, store_costs, store_theta

def main():
    args_length = len(sys.argv);
    if(args_length<2):
        print("Insufficient arguments provided")
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
    # print(x1, x1.mean(), x1.var(), x1.std())
    theta_sample = [3, 1, 2]
    # Q2 Part (A) Generating Sample Data
    samples = pd.DataFrame();
    samples[0] = x1
    samples[1] = x2
    samples[2] = 1
    samples = samples[[2, 0, 1]]
    data_x = samples.to_numpy()
    print(data_x)
    samples[3] = theta_sample[0]*1 + theta_sample[1]*x1 + theta_sample[2]*x2 + noise;
    all_data = samples.to_numpy()
    print(all_data)
    
    # Q2 Part (B) Stochastic Gradient Descent
    result_list_b1 = stochastic(all_data, 0.001, 1, 10000, 1000000, 1e-4)
    result_list_b2 = stochastic(all_data, 0.001, 100, 10000, 1000000, 1e-4)
    # result_list_b3 = stochastic(all_data, 0.001, 10000, 10000, 1000000, 1e-4)
    # result_list_b4 = stochastic(all_data, 0.001, 1000000, 30000, 1000000, 1e-4)
    # print(result_list_b1[0], result_list_b2[0], result_list_b3[0], result_list_b4[0])

    fig = plt.figure(1)
    ax = plt.axes(projection='3d')
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("theta2")
    axes = plt.gca()
    ax.set_title("theta with batch size: 1")
    theta_history1 = result_list_b1[2]
    theta_history1 = np.array(theta_history1)
    theta_history1 = theta_history1.reshape(theta_history1.shape[0], 3)
    theta0 = theta_history1[:, 0:1]
    theta1 = theta_history1[:, 1:2]
    theta2 = theta_history1[:, 2:3]

    theta0 = np.array(theta0).reshape(theta_history1.shape[0])
    theta1 = np.array(theta1).reshape(theta_history1.shape[0])
    theta2 = np.array(theta2).reshape(theta_history1.shape[0])

    ax.set_xlim([np.min(theta0), np.max(theta0)])
    ax.set_ylim([np.min(theta1), np.max(theta1)])
    ax.set_zlim([np.min(theta2), np.max(theta2)])
    ax.scatter(theta0, theta1, theta2)

    plt.savefig("theta_with_batch_size_1.png")

    fig = plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.set_xlabel("theta0")
    ax.set_ylabel("theta1")
    ax.set_zlabel("theta2")
    axes = plt.gca()
    ax.set_title("theta with batch size 100")
    theta_history2 = result_list_b2[2]
    theta_history2 = np.array(theta_history2)
    theta_history2 = theta_history2.reshape(theta_history2.shape[0], 3)
    theta0 = theta_history2[:, 0:1]
    theta1 = theta_history2[:, 1:2]
    theta2 = theta_history2[:, 2:3]

    theta0 = np.array(theta0).reshape(theta_history2.shape[0])
    theta1 = np.array(theta1).reshape(theta_history2.shape[0])
    theta2 = np.array(theta2).reshape(theta_history2.shape[0])

    ax.set_xlim([np.min(theta0), np.max(theta0)])
    ax.set_ylim([np.min(theta1), np.max(theta1)])
    ax.set_zlim([np.min(theta2), np.max(theta2)])
    ax.scatter(theta0, theta1, theta2)

    plt.savefig("theta_with_batch_size_100.png")
        


if __name__ == "__main__":
    main()
