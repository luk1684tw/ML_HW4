import argparse

import numpy as np


def generate_cluster(mean1, var1, mean2, var2, num_data):
    data_x = np.random.normal(mean1, var1, num_data)
    data_y = np.random.normal(mean2, var2, num_data)

    return np.vstack((data_x, data_y)).T



def gradient_descent(w, a, d):
    pass



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_data", type=int, help="number of data points in logistic regression")
    parser.add_argument("-T", "--theda", type=int, nargs="+", help="parameters of clusters")


    args = parser.parse_args()
    D1 = generate_cluster(args.theda[0], args.theda[1], args.theda[2], args.theda[3],  args.num_data)
    D2 = generate_cluster(args.theda[4], args.theda[5], args.theda[6], args.theda[7],  args.num_data)
    D = np.concatenate((D1, D2))
    W_gradient, W_newton = np.random.uniform(size=3), np.random.uniform(size=3)  # shape (1X3)

    X = list()
    for dp in D:
        d = list(np.concatenate((np.array([1]), dp)))
        X.append(d)
    X = np.array(X) # shape (num_dataX3)
    Y =  np.array([1] * args.num_data + [2] * args.num_data)

    step = np.dot(X.T, (Y - 1 / (1 + np.exp(-np.dot(X, W_gradient))))) #steepest gradient descent
    W_gradient_pos = W_gradient + step
    while not np.allclose(W_gradient, W_gradient_pos, atol=0.001):
        step = np.dot(X.T, (Y - 1 / (1 + np.exp(-np.dot(X, W_gradient)))))
        print (step)
        W_gradient_pos, W_gradient = W_gradient + step, W_gradient_pos

