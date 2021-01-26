import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    labels = [0,1,2,3,4,5,6,7,8,9]

    # permuting the images and labels
    indices = np.arange(im_train.shape[1])
    np.random.shuffle(indices)

    im_train = im_train.T
    label_train = label_train.T

    shuffled_im = im_train[indices]
    shuffled_label = label_train[indices]

    shuffled_im = shuffled_im.T
    shuffled_label=shuffled_label.T

    i = 0
    mini_batch_x = []
    mini_batch_y = []

    while i < shuffled_im.shape[1]:
        temp_x = []
        temp_y = []
        for j in range(i, i+batch_size):
            if j == shuffled_im.shape[1]:
                break
            else:
                temp_x.append(shuffled_im[:, j])
                # one hot encoding
                temp_label = np.zeros(10)
                temp_label[labels.index(shuffled_label[0, j])] = 1
                temp_y.append(temp_label)
        i += batch_size

        mini_batch_x.append(np.array(temp_x))
        mini_batch_y.append(np.array(temp_y))

    mini_batch_x = np.array(mini_batch_x)
    mini_batch_y = np.array(mini_batch_y)
    return mini_batch_x, mini_batch_y


def fc(x, w, b):
    y = w @ x + b
    return y

def fc_backward(dl_dy, x, w, b, y):
    n = dl_dy.shape[1]
    m = x.shape[0]

    dl_dx = dl_dy @ w

    dy_dw = np.zeros((n, n, m))
    for i in range(n):
        dy_dw[i][i] = x.T
    dl_dw = dl_dy @ dy_dw

    dl_db = dl_dy.T
    return dl_dx, dl_dw, dl_db

def loss_euclidean(y_tilde, y):
    l = np.linalg.norm(y-y_tilde)
    dl_dy = -2*(y-y_tilde).T
    return l, dl_dy

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def loss_cross_entropy_softmax(x, y):
    smax = softmax(x)
    log_softmax = np.log(smax)
    l = np.sum(np.multiply(y, log_softmax))/10
    dl_dy = softmax(x) - y
    dl_dy = np.reshape(dl_dy, (1,10))
    return l, dl_dy

def relu(x):
#     y = np.maximum(x, 0)
    # leaky
    y = np.maximum(x, .01*x)
    return y


def relu_backward(dl_dy, x, y):
    dy_dx = x > 0
    dy_dx = dy_dx.astype(int)
    # leaky
    dy_dx[dy_dx < 0] = .01

    dl_dx = dl_dy * dy_dx.T

    return dl_dx

def im2col(x,hh,ww):
    h, w, c = x.shape
    # if x is padded, new_h should equal h, new_w should equal w
    new_h = (h-hh) + 1
    new_w = (w-ww) + 1
    col = np.zeros([new_h*new_w,c*hh*ww])

    for i in range(new_h):
        for j in range(new_w):
            patch = x[i:i+hh,j:j+ww, ...]
            col[i*new_w+j,:] = np.reshape(patch,-1)
    return col.T

def reshape_weights(w):
    h, width, c1, c2 = w.shape
    new = np.reshape(w, ((h*width*c1), c2))
    return new.T

def conv(x, w_conv, b_conv):
    H, W, c1 = x.shape
    h, w, c1, c2 = w_conv.shape

    pad_x = np.lib.pad(x, ((1,1),(1,1),(0,0)),'constant', constant_values=0)
    y = np.zeros((H, W, c2))

    reshaped_w = reshape_weights(w_conv) # 3x9
    reshaped_inp = im2col(pad_x, h, w) # 9x196
    product = reshaped_w @ reshaped_inp # 3x196
    h, w = product.shape
    for i in range(h): # adding bias
        product[i, :] += b_conv[i, 0]
    y = np.reshape(product, (14,14,3)) # reshaping back to normal
    return y

def conv_alternate(x, w_conv, b_conv):
    H, W, c1 = x.shape
    h, w, c1, c2 = w_conv.shape

    pad_x = np.lib.pad(x, ((1,1),(1,1),(0,0)),'constant', constant_values=0)
    y = np.zeros((H, W, c2))

    for N in range(c2):
        for depth in range(c1):
            for i in range(H):
                for j in range(W):
                    y[i, j, N] = np.sum(pad_x[i:i+h, j:j+w, depth] * w_conv[:,:,depth,N]) + b_conv[N][0]
    return y

def conv_backward(dl_dy, x, w_conv, b_conv, y):
    h, w, c1 = x.shape
    hh, ww, c1, c2 = w_conv.shape

    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)
    dl_db = np.sum(dl_dy, axis=(0,1)) # summing over rows and columns

    reshaped_dy = np.reshape(dl_dy, (3, 196))

    pad_x = np.lib.pad(x, ((1,1),(1,1),(0,0)),'constant', constant_values=0)
    im_col = im2col(pad_x, hh, ww) # 9 x 196

    product = reshaped_dy @ im_col.T
    dl_d2 = np.reshape(product, (3,3,1,3))
    return dl_dw, dl_db

def conv_backward_alternate(dl_dy, x, w_conv, b_conv, y):
    h, w, c1 = x.shape
    hh, ww, c1, c2 = w_conv.shape

    dl_dw = np.zeros_like(w_conv)
    dl_db = np.zeros_like(b_conv)
    dl_db = np.sum(dl_dy, axis=(0,1))

    pad_x = np.lib.pad(x, ((1,1),(1,1),(0,0)),'constant', constant_values=0)

    for a in range(c2):
        for i in range(hh):
            for j in range(ww):
                for k in range(h):
                    for l in range(w):
                        for b in range(c1):
                            dl_dw[i, j, b, a] += pad_x[i+k, j+l, b] * dl_dy[k, l, a]

    return dl_dw, dl_db

def pool2x2(x):
    h, w, c = x.shape
    y = np.zeros((h//2, w//2, c))
    for i in range(h//2):
        for j in range(w//2):
            for k in range(c):
                y[i,j,k] = np.amax(x[i*2:i*2+2, j*2:j*2+2,k])
    return y

def pool2x2_backward(dl_dy, x, y):
    dl_dx = np.zeros_like(x)
    h,w,c = x.shape

    for i in range(h//2):
        for j in range(w//2):
            for k in range(c):
                temp = np.amax(x[i*2:i*2+2, j*2:j*2+2,k])
                for m in range(2):
                    for n in range(2):
                        if x[i*2+m, j*2+n, k] == temp and temp != 0:
                            dl_dx[i*2+m, j*2+n, k] = dl_dy[i, j, k]
                        else:
                            dl_dx[i*2+m, j*2+n, k] = 0
    return dl_dx


def flattening(x):
    return x.flatten('F').reshape((x.size, 1))


def flattening_backward(dl_dy, x, y):
    h, w, c = x.shape
    dl_dx = np.reshape(dl_dy, (h, w, c), 'F')
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    learn_rate = .1
    num_iterations = 5000
    decay_rate = .1
    weights = np.random.randn(10, 196)
    bias = np.zeros((10,1))
    k = 0
    for i in range(num_iterations):
        if i % 1000 == 0:
            learn_rate = learn_rate*decay_rate
            print(learn_rate)
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))
        for j in range(mini_batch_x[k].shape[0]):
            inp = np.reshape(mini_batch_x[k][j],(196,1))
            predict = fc(inp, weights, bias)
            loss, dl_dy = loss_euclidean(predict, np.reshape(mini_batch_y[k][j], (10,1)))
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, inp, weights, bias, predict)
            dL_dw = dL_dw + dl_dw[:,0,:]
            dL_db = dL_db + dl_db
        k += 1
        if k > mini_batch_x[k].shape[0]:
            k = 0
        weights = weights - learn_rate*dL_dw/32
        bias = bias - learn_rate*dL_db/32

    return weights, bias

def train_slp(mini_batch_x, mini_batch_y):
    learn_rate = .5
    num_iterations = 5000
    decay_rate = .9
    weights = np.random.randn(10, 196)
    bias = np.zeros((10,1))
    k = 0
    losses = []
    for i in range(num_iterations):
        if i % 1000 == 0 and i > 0:
            learn_rate = learn_rate*decay_rate
            print(learn_rate)
        dL_dw = np.zeros((10, 196))
        dL_db = np.zeros((10, 1))
        for j in range(mini_batch_x[k].shape[0]):
            inp = np.reshape(mini_batch_x[k][j],(196,1))
            predict = fc(inp, weights, bias)
            loss, dl_dy = loss_cross_entropy_softmax(predict, np.reshape(mini_batch_y[k][j], (10,1)))
            losses.append(loss)
            dl_dx, dl_dw, dl_db = fc_backward(dl_dy, inp, weights, bias, predict)
            dL_dw = dL_dw + dl_dw[:,0,:]
            dL_db = dL_db + dl_db
        k += 1
        if k > mini_batch_x[k].shape[0]:
            k = 0
        weights = weights - learn_rate*dL_dw/32
        bias = bias - learn_rate*dL_db/32

    return weights, bias

def train_mlp(mini_batch_x, mini_batch_y):
    learn_rate = .5
    num_iterations = 5000
    decay_rate = .9
    w1 = np.random.randn(30, 196)
    b1 = np.zeros((30,1))
    w2 = np.random.randn(10, 30)
    b2 = np.zeros((10,1))
    k = 0
    for i in range(num_iterations):
        if i % 1000 == 0 and i > 0:
            learn_rate = learn_rate*decay_rate
            print(learn_rate)
        dL_dw2 = np.zeros((10, 30))
        dL_db2 = np.zeros((10, 1))
        dL_dw1 = np.zeros((30, 196))
        dL_db1 = np.zeros((30, 1))
        for j in range(mini_batch_x[k].shape[0]):
            inp = np.reshape(mini_batch_x[k][j],(196,1))
            predict1 = fc(inp, w1, b1)
            activated = relu(predict1)
            predict2 = fc(activated, w2, b2)
            loss, dl_dy2 = loss_cross_entropy_softmax(predict2, np.reshape(mini_batch_y[k][j], (10,1)))
            dl_dx2, dl_dw2, dl_db2 = fc_backward(dl_dy2, activated, w2, b2, predict2)
            dl_dy1 = relu_backward(dl_dx2, predict1, activated)
            dl_dx1, dl_dw1, dl_db1 = fc_backward(dl_dy1, inp, w1, b1, predict1)
            dL_dw2 = dL_dw2 + dl_dw2[:,0,:]
            dL_db2 = dL_db2 + dl_db2
            dL_dw1 = dL_dw1 + dl_dw1[:,0,:]
            dL_db1 = dL_db1 + dl_db1
        k += 1
        if k > mini_batch_x[k].shape[0]:
            k = 0
        w1 = w1 - learn_rate*dL_dw1/32
        b1 = b1 - learn_rate*dL_db1/32
        w2 = w2 - learn_rate*dL_dw2/32
        b2 = b2 - learn_rate*dL_db2/32
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    learn_rate = 1
    num_iterations = 2000
    decay_rate = .1
    w_conv = np.random.randn(3,3,1,3)
    b_conv = np.zeros((3,1))
    w_fc = np.random.randn(10, 147)
    b_fc = np.zeros((10,1))
    k = 0
    losses = []
    for i in range(num_iterations):
        if i % 1000 == 0 and i > 0:
            learn_rate = learn_rate*decay_rate
            print(learn_rate)
        dL_dw_conv = np.zeros((3,3,1,3))
        dL_db_conv = np.zeros((3, 1))
        dL_dw_fc = np.zeros((10, 147))
        dL_db_fc = np.zeros((10, 1))
        for j in range(mini_batch_x[k].shape[0]):
            inp = np.reshape(mini_batch_x[k][j],(14,14,1))
            predict1 = conv(inp, w_conv, b_conv)
            activated = relu(predict1)
            pooled = pool2x2(activated)
            flat = flattening(pooled)
            predict2 = fc(flat, w_fc, b_fc)
            loss, dl_dy = loss_cross_entropy_softmax(predict2, np.reshape(mini_batch_y[k][j], (10,1)))
            losses.append(loss)
            dl_dy, dl_dw_fc, dl_db_fc = fc_backward(dl_dy, flat, w_fc, b_fc, predict2)
            dl_dy = flattening_backward(dl_dy, pooled, flat)
            dl_dy = pool2x2_backward(dl_dy, activated, pooled)
            dl_dy = relu_backward(dl_dy.T, predict1, activated)
            dl_dw_conv, dl_db_conv = conv_backward(dl_dy.T, inp, w_conv, b_conv, predict1)

            dL_dw_conv = dL_dw_conv + dl_dw_conv
            dL_db_conv = dL_db_conv + dl_db_conv
            dL_dw_fc = dL_dw_fc + dl_dw_fc[:,0,:]
            dL_db_fc = dL_db_fc + dl_db_fc

        k += 1
        if k > mini_batch_x[k].shape[0]:
            k = 0
        w_conv = w_conv - learn_rate*dL_dw_conv/32
        b_conv = b_conv - learn_rate*dL_db_conv/32
        w_fc = w_fc - learn_rate*dL_dw_fc/32
        b_fc = b_fc - learn_rate*dL_db_fc/32

    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    print("done1")
    main.main_slp()
    print("done2")
    main.main_mlp()
    print("done3")
    main.main_cnn()
    print("done4")
