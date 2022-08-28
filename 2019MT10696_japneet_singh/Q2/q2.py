from math import sqrt
import pandas as pd
import sys
import numpy as np

def cost(theta, samples, batch_size, batch_number):
    aggregate_error = 0
    start_index = (batch_number-1)*batch_size 
    for i in range(0, batch_size):
        aggregate_error += (samples[2][i] - theta[0] - theta[1]*samples[0][i] - theta[2]*samples[1][i])**2;
    return aggregate_error/(2);

def lms_update_theta(samples, batch_number, batch_size, theta, index):
    start_index = (batch_number-1)*batch_size
    aggregate_error = 0
    for i in range(0, batch_size):
        xval = 1
        if(index==1 or index==2):
            xval=samples[index-1][i + start_index]
        aggregate_error += (samples[2][i + start_index] - theta[0]*1 - theta[1]*samples[1][i + start_index] - theta[1]*samples[1][i + start_index])*xval;
    return aggregate_error;

def stochastic_gradient_descent(batch_size, samples, theta):
    # print(samples[0])
    samples = samples.sample(frac=1).reset_index()
    # print(samples[0])
    learning_rate = 0.001
    stopping_criteria = 100
    batch_number = 0
    batches = samples.size/batch_size
    while True:
        if(cost(theta, samples, batch_size, batch_number)<stopping_criteria):
            return theta
        print(theta, cost(theta, samples, batch_size, batch_number))
        batch_number += 1
        theta_zero = theta[0] + learning_rate*(lms_update_theta(samples, (batch_number%batches), batch_size, theta, 0))
        theta_one = theta[1] + learning_rate*(lms_update_theta(samples, (batch_number%batches), batch_size, theta, 1))
        theta_two = theta[2] + learning_rate*(lms_update_theta(samples, (batch_number%batches), batch_size, theta, 2))
        theta = [theta_zero, theta_one, theta_two]

def main():
    args_length = len(sys.argv);
    if(args_length<2):
        print("Insufficient arguments provided")
    path_test_data = sys.argv[1]
    mu1 = 3
    sigma1 = 4
    mu2 = -1
    sigma2 = 4
    mu_noise = 0
    sigma_noise = 2
    x1 = np.random.normal(mu1, sqrt(sigma1), 1000000);
    x2 = np.random.normal(mu2, sqrt(sigma2), 1000000);
    noise = np.random.normal(mu_noise, sqrt(sigma_noise), 1000000);
    # print(x1, x1.mean(), x1.var(), x1.std())
    theta = [0, 0, 0]
    samples = pd.DataFrame();
    samples[0] = x1
    samples[1] = x2
    samples[2] = 3 + 1*x1 + 2*x2 + noise;
    theta = stochastic_gradient_descent(100, samples, theta)

if __name__ == "__main__":
    main()
