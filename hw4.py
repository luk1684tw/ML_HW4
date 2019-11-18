import argparse
import struct

import numpy as np
import numba as nb

import matplotlib.pyplot as plt

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


def print_part1(result, num_data, w, method):
    print (f"{method}:\n")
    print ("W:")
    print (w)
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(2 * num_data):
        if Y[i] == 0 and result[i] == 0:
            tp += 1
        elif Y[i] == 0 and result[i] == 1:
            fn += 1
        elif Y[i] == 1 and result[i] == 0:
            fp += 1
        else:
            tn += 1
    print ("confusion matrix:")
    print ("                Predict cluster1      Predict cluster2")
    print(f"  Is cluster 1               {tp}                   {fn}")
    print(f"  Is cluster 2               {fp}                   {tn}")
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    print ("Sensitivity: ", sensitivity)
    print ("Sepcificity: ", specificity)

    return


def plot_image(data_points, predict_gradient, predict_newton, num_data):
    fig, ax = plt.subplots(1, 3)

    ax[0].plot(data_points[:num_data, 0], data_points[:num_data, 1], 'bo')
    ax[0].plot(data_points[num_data:, 0], data_points[num_data:, 1], 'ro')
    ax[0].set_title("Ground Truth")

    cluster1, cluster2 = list(), list()

    for i in range(2 * num_data):
        if predict_gradient[i] == 0:
            cluster1.append(data_points[i])
        else:
            cluster2.append(data_points[i])
    
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    ax[1].plot(cluster1[:, 0], cluster1[:, 1], 'bo')
    ax[1].plot(cluster2[:, 0], cluster2[:, 1], 'ro')
    ax[1].set_title("Gradient descent")

    cluster1, cluster2 = list(), list()
    for i in range(2 * num_data):
        if predict_newton[i] == 0:
            cluster1.append(data_points[i])
        else:
            cluster2.append(data_points[i])
    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)
    ax[2].plot(cluster1[:, 0], cluster1[:, 1], 'bo')
    ax[2].plot(cluster2[:, 0], cluster2[:, 1], 'ro')
    ax[2].set_title("Newton's method")

    plt.savefig("part1.png")

    return


def read_mnist(image_path, label_path):
    with open(label_path, 'rb') as label:
        magic, n = struct.unpack('>II', label.read(8))
        labels = np.fromfile(label, dtype=np.uint8)
    with open(image_path, 'rb') as image:
        magic, num, rows, cols = struct.unpack('>IIII', image.read(16))
        images = np.fromfile(image, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


@nb.jit()
def e_step(p, lamb, train_images):
    w = np.ones((60000, 10), dtype=np.float64)
    for i in range(60000):
        for idx, pixel in enumerate(train_images[i]):
            if pixel == 1:
                w[i, :] *= p[:, idx]
            else:
                w[i, :] *= (1 - p[:, idx])
        w[i] *= lamb
        w[i] /= np.sum(w[i])

    return w


@nb.jit()
def m_step(w, train_images):
    p = np.zeros((10, 784))
    lamb = np.sum(w, axis=0) / len(train_images)
    
    for i in range(10):
        for pixel in range(784):  
            x = train_images[:, pixel] == 1
            nominator = np.sum(w[x, i])

            p[i, pixel] = nominator / np.sum(w[:, i])

    return lamb, p


def plot_mnist(p):
    paras = np.copy(p)
    paras.resize(10, 28, 28)
    for idx, cluster in enumerate(paras):
        print (f"class {idx}")
        for i in cluster:
            line = ""
            for j in i:
                if j >= 0.5:
                    line += "1"
                else:
                    line += "0"
            print (f"{line}\n")
        print ('------------------------------')

    return


def em_algorithm():
    global lamb, p, train_images, iteraton, pre_p
    w = e_step(p, lamb, train_images)
    lamb, p = m_step(w, train_images)
    plot_mnist(p)
    iteraton += 1
    tmp = np.sum(np.abs(p - pre_p), axis=0)
    difference = np.sum(tmp)
    print (f"No. Iter{iteraton}, Difference: {difference}" )

    return


def make_confusion(result_mnist, label_mapping, train_labels):
    for label, cluster in enumerate(label_mapping):
        tp, fp, fn, tn = 0, 0, 0, 0
        for i in range(60000):
            if result_mnist[i] == cluster and train_labels[i] == label:
                tp += 1
            elif result_mnist[i] == cluster and train_labels[i] != label:
                fp += 1
            elif result_mnist[i] != cluster and train_labels[i] == label:
                fn += 1
            else:
                tn += 1
        print(f"confusion matrix of class{label}:")
        print(f"                Predict class{label}    Predict not class{label}")
        print(f"  Is  class {label}               {tp}                   {fn}")
        print(f"  Not class {label}               {fp}                   {tn}")
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        print ("Sensitivity: ", sensitivity)
        print ("Sepcificity: ", specificity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--num_data", type=int, help="number of data points in logistic regression")
    parser.add_argument("-T", "--theda", type=int, nargs="+", help="parameters of clusters")


    args = parser.parse_args()
    D1 = generate_cluster(args.theda[0], args.theda[1], args.theda[2], args.theda[3],  args.num_data)
    D2 = generate_cluster(args.theda[4], args.theda[5], args.theda[6], args.theda[7],  args.num_data)
    D = np.concatenate((D1, D2))
    W_gradient, W_newton = np.random.uniform(size=3), np.random.uniform(size=3)
    lr = 0.2

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
    predict_gradient = [0 if x < 0.5 else 1 for x in res_gradient]
    # print_part1(predict_gradient, args.num_data, W_gradient, "Gradient descent")

    print ("\n---------------------------")

    res_newton = 1 / (1 + np.exp(-1 * np.dot(X, W_newton)))
    predict_newton = [0 if x < 0.5 else 1 for x in res_newton]
    
    # print_part1(predict_newton, args.num_data, W_newton, "Newton's method")
    plot_image(D, predict_gradient, predict_newton, args.num_data)

    # part2
    train_image = './train-images-idx3-ubyte'
    train_label = './train-labels-idx1-ubyte'

    train_images, train_labels = read_mnist(train_image, train_label)
    lamb = np.random.uniform(0.05, 0.15, 10)
    p = np.random.uniform(0.35, 0.65, 10*784)
    p.resize(10, 784)
    pre_p = np.copy(p)
    train_images = train_images // 128
    iteraton = 0
    em_algorithm()
    
    while not np.allclose(p, pre_p, atol=0.001):
        pre_p = np.copy(p)
        em_algorithm()

    predict_mnist = e_step(p, lamb, train_images)
    
    result_mnist = np.argmax(predict_mnist, axis=1)
    cluster_label = np.zeros((10, 10))
    label_mapping = np.zeros(10) - 1

    for i, prediction in enumerate(result_mnist):
        cluster_label[prediction, train_labels[i]] += 1

    for i in range(10):
        label = np.amax(cluster_label)
        x, y = np.where(cluster_label == label)
        label_mapping[y] = x
        for i in range(10):
            if cluster_label[x, i] > 0:
                cluster_label[x, i] *= -1
            if cluster_label[i, y] > 0:
                cluster_label[i, y] *= -1
        if cluster_label[x, y] > 0:
            cluster_label[x, y] *= -1


    plot_mnist(p)
    make_confusion(result_mnist, label_mapping, train_labels)
    print (label_mapping)
