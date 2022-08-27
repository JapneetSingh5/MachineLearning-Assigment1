from math import sqrt
import numpy as np
import sys
from os import listdir
import pandas as pd
import matplotlib.pyplot as plt 

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
    a = 0
    b = 0
    learning_rate = 0.001
    stopping_criteria = 1e-10
    while True:
        a_new = a + learning_rate*(lms_update_a(x,y,a,b))
        b_new = b + learning_rate*(lms_update_b(x,y,a,b))
        if(sqrt((a_new - a)**2 + (b_new - b)**2)<stopping_criteria):
            return (a_new,b_new)
        a = a_new
        b = b_new
        print(a_new, b_new)




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
    (a,b) = batch_gradient_descent(df[0], df[1])
    print(a,b)
    print(average_error(df[0],df[1],a,b))
    plt.scatter(df[0], df[1])
    plt.scatter(df[0], df[0].transform(lambda x: a*x + b))
    plt.show()
    


if __name__ == "__main__":
    main()


