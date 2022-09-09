from math import sqrt,exp,log
from numbers import Number
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt 

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
    test_x_file_extension = '/X.csv'
    train_y_file_extension = '/q4y.dat'
    test_y_file_extension = '/result_4.txt'
    args_length = len(sys.argv);
    if(args_length<3):
        print("Insufficient arguments provided, exiting")
        sys.exit(1)
    path_train_data = sys.argv[1]
    path_test_data = sys.argv[2]
    train_data_x = path_train_data + train_x_file_extension;
    test_data_x = path_test_data + test_x_file_extension;
    train_data_y = path_train_data + train_y_file_extension;
    test_result_y = path_test_data + test_y_file_extension
    # Step 2. Prepare input data
    df = pd.read_csv(train_data_x, header=None,sep='  ',engine='python')
    # 2(i) Normalize data to zero mean, unit variance
    # print("np.mean", np.mean(df.to_numpy()), "np.std", np.std(df.to_numpy()))
    # NumPy giving more precise mean values than Pandas, hence used its mean function
    x1_mean = np.mean(df[0].to_numpy())
    x1_std = np.std(df[0].to_numpy())
    df[0] = (df[0] - x1_mean)/x1_std;
    # print(df[0].mean(), df[0].std(),np.mean(df[0].to_numpy()), np.std(df[0].to_numpy()))
    # 2(ii) Normalize X2
    x2_mean = np.mean(df[1].to_numpy())
    x2_std = np.std(df[1].to_numpy())
    df[1] = (df[1] - x2_mean)/x2_std;
    print(df[1].mean(), df[1].std())
    # 2(ii) build NumPy array for further usage
    X = df.to_numpy()
    # 2(iv) build y values NumPy array
    df[3] = pd.read_csv(train_data_y, header=None)
    Y = df[3].to_numpy().reshape(-1,1)
    Y_bin = np.array((Y=='Canada'), dtype=np.float)
    # print(Y_bin)
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

    plt.contour(x1, x2, z, levels=[0])
    plt.title("Training Data with learned boundary")
    plt.savefig("LinearBoundary.png")

    D = log(phi/(1-phi)) - 0.5*log(abs(np.linalg.det(sigma_one))/abs(np.linalg.det(sigma_zero))) 
    for i in range(0, x1.shape[0]):
        for j in range(0, len(z[0])):
            x = np.array([x1[i][j], x2[i][j]])
            x = x.reshape(2, 1)
            z[i][j] = D + 0.5*np.dot((x-u_zero).T, np.matmul(sigma_zero_inv, (x-u_zero))) - 0.5*np.dot((x-u_one).T, np.matmul(sigma_one_inv, (x-u_one)))

    plt.contour(x1, x2, z, levels=[0])
    plt.title("Training Data with learned boundary")
    plt.savefig("QuadraticBoundary.png")


    # plt.show()
    # plt.close()

    df_test = pd.read_csv(test_data_x, header=None)
    df_test[0] = (df_test[0] - x1_mean)/x1_std;
    df_test[1] = (df_test[1] - x2_mean)/x2_std;
    # df_test = df_test[[1, 0]]
    X_test = df_test.to_numpy()
    print(X_test)
    Y_result = [ ( log(phi/(1-phi)) - 0.5*log(abs(np.linalg.det(sigma_one))/abs(np.linalg.det(sigma_zero))) + 0.5*np.dot((row.reshape(2,1)-u_zero).T, np.matmul(sigma_zero_inv, (row.reshape(2,1)-u_zero))) - 0.5*np.dot((row.reshape(2,1)-u_one).T, np.matmul(sigma_one_inv, (row.reshape(2,1)-u_one))) ) for row in X_test]
    Y_result = [int(x>=0) for x in Y_result]
    Y_category = np.array(["SampleCategory"]*len(Y_result))
    for i in range(0, Y_category.shape[0]):
        if Y_result[i]==1 :
            Y_category[i] = 'Canada'
        else:
            Y_category[i] =  'Alaska'
    df_res = pd.DataFrame()
    df_res[0] = pd.array(Y_category)
    df_res.to_csv(test_result_y, header=None, index=False)



if __name__ == "__main__":
    main()


