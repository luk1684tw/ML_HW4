import argparse

import numpy as np


def generate_cluster(mean1, var1, mean2, var2, num_data):
    data_x = np.random.normal(mean1, var1, num_data)
    data_y = np.random.normal(mean2, var2, num_data)

    return np.vstack((data_x, data_y)).T



def gradient_descent(W, X, Y):
    step = np.dot(X.T, (Y - 1 / (1 + np.exp(-np.dot(X, W)))))
    
    return step


def hassian_matrix(W, X, n):
    T = np.zeros((n, n))
    for i in range(n):
        exponential = np.exp(-1 * np.sum(X[i] * W))
        if np.isinf(exponential):
            print ("random select exp")
            exponential = np.exp(np.random.randint(1, 700))
            while np.isinf(exponential**2):
                print ("reselect exp")
                exponential = np.exp(np.random.randint(1, 700))
            print (exponential)
        
        T[i, i] = exponential / (1 + exponential)**2

    return np.dot(np.dot(X.T, T), X)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_data", type=int, help="number of data points in logistic regression")
    parser.add_argument("-T", "--theda", type=int, nargs="+", help="parameters of clusters")


    args = parser.parse_args()
    D1 = generate_cluster(args.theda[0], args.theda[1], args.theda[2], args.theda[3],  args.num_data)
    D2 = generate_cluster(args.theda[4], args.theda[5], args.theda[6], args.theda[7],  args.num_data)
    D = np.concatenate((D1, D2))
    W_gradient, W_newton = np.random.uniform(size=3), np.random.uniform(size=3)
    lr = 0.1

    X = list()
    for dp in D:
        d = list(np.concatenate((np.array([1]), dp)))
        X.append(d)
    X = np.array(X) # shape (num_dataX3)
    Y =  np.array([0] * args.num_data + [1] * args.num_data)

    # part 1-gradient 
    step = gradient_descent(W_gradient, X, Y)
    W_gradient_pos = np.copy(W_gradient + step)
    counter = 0
    while not np.allclose(W_gradient, W_gradient_pos, atol=0.01):
        W_gradient = np.copy(W_gradient_pos)
        W_gradient_pos += lr * gradient_descent(W_gradient, X, Y)
        if counter > 100000:
            break

        counter += 1

    # part 1-newton
    W_newton_pos = np.random.uniform(size=3)
    counter = 0
    while not np.allclose(W_newton, W_newton_pos, atol=0.01):
    # for i in range(1):
        if counter > 100000:
            break
        
        counter += 1
        hessian = hassian_matrix(W_newton, X, args.num_data * 2)
        gradient = gradient_descent(W_newton, X, Y)
        try:
            h = np.linalg.inv(hessian)
            step = np.dot(h, gradient)
        except np.linalg.LinAlgError as err:
            print ('Singular Hessian')
            step = gradient
        W_newton_pos = np.copy(W_newton)
        W_newton = np.copy(W_newton + step*0.1)


    res_gradient = 1 / (1 + np.exp(-1 * np.dot(X, W_gradient)))
    predict_gradient = [1 if x < 0.5 else 2 for x in res_gradient]
    print (predict_gradient)

    res_newton = 1 / (1 + np.exp(-1 * np.dot(X, W_newton)))
    predict_newton = [1 if x < 0.5 else 2 for x in res_newton]
    print (predict_newton)

