from math import sqrt
import pandas as pd
import sys
import numpy as np

def cost(theta, samples_batch):
    aggregate_error = 0
    for i in range(0, len(samples_batch)):
        aggregate_error += (samples_batch[2][i] - theta[0] - theta[1]*samples_batch[0][i] - theta[2]*samples_batch[1][i])**2;
    return aggregate_error/(2*len(samples_batch));

def lms_update_theta(samples_batch, theta, index):
    aggregate_error = 0
    for i in range(0, len(samples_batch)):
        xval = 1
        if(index==1 or index==2):
            xval=samples_batch[index-1][i]
        aggregate_error += (samples_batch[2][i] - theta[0]*1 - theta[1]*samples_batch[1][i] - theta[1]*samples_batch[1][i])*xval;
    return aggregate_error;

def stochastic_gradient_descent(batch_size, batch_count, samples, theta_init, learning_rate, stopping_criteria, max_iterations):
    # print(samples[0])
    # print(samples[0])
    return_list = []
    batch_number = 0
    theta = theta_init
    counter = 0
    while True:
        counter+=1;
        start_index = batch_number*batch_size
        end_index = batch_number*batch_size + batch_size
        samples_batch = samples.iloc[start_index:end_index].copy()
        samples_batch = samples_batch.reset_index(drop=True)
        print(samples_batch)
        batch_number += 1
        batch_number = batch_number % batch_count
        theta_zero = theta[0] + learning_rate*(lms_update_theta(samples_batch, theta, 0))
        theta_one = theta[1] + learning_rate*(lms_update_theta(samples_batch, theta, 1))
        theta_two = theta[2] + learning_rate*(lms_update_theta(samples_batch, theta, 2))
        theta = [theta_zero, theta_one, theta_two]
        print(theta)
        current_cost = cost(theta, samples_batch);
        return_list.append([theta, current_cost])
        print(theta, current_cost)
        if(current_cost<stopping_criteria or counter>max_iterations):
            return return_list


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
    samples[2] = theta_sample[0]*1 + theta_sample[1]*x1 + theta_sample[2]*x2 + noise;
    
    # Q2 Part (B) Stochastic Gradient Descent
    
    # (i) Batch Size = 1
    batch_size = 1000000
    learning_rate = 0.0001
    stopping_criteria = 2*1e-5
    batch_count = round(1000000/batch_size)
    theta_init = [0, 0, 0]
    samples = samples.sample(frac=1).reset_index(drop=True)
    max_iterations = 10000
    theta_list = stochastic_gradient_descent(batch_size, batch_count, samples, theta_init, learning_rate, stopping_criteria, max_iterations)
    


if __name__ == "__main__":
    main()
