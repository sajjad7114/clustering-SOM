from random import random
from matplotlib import pyplot as plt
import numpy as np
import cv2


def find_bmu(s_mat, vec):
    a, b, c = s_mat[0][0]
    m = (a-vec[0])**2 + (b-vec[1])**2 + (c-vec[2])**2
    ii = 0
    jj = 0
    for i in range(len(s_mat)):
        for j in range(len(s_mat[i])):
            a, b, c = s_mat[i][j]
            d = (a-vec[0])**2 + (b-vec[1])**2 + (c-vec[2])**2
            if d < m:
                m = d
                ii = i
                jj = j
    return ii, jj


def update_som(s_mat, ii, jj, vec, alpha, count):
    for i in range(len(s_mat)):
        for j in range(len(s_mat[i])):
            d = ((i-ii)**2 + (j-jj)**2)**0.5
            if d < radius(count):
                update_cell(s_mat, i, j, vec, alpha * N(d, count))


def update_cell(s_mat, ic, jc, vec, alpha):
    a, b, c = s_mat[ic][jc]
    a = a + alpha * (vec[0]-a)
    b = b + alpha * (vec[1]-b)
    c = c + alpha * (vec[2]-c)
    s_mat[ic][jc] = (a, b, c)


def L(alpha, count):
    return alpha * 0.75**count


def N(d, count):
    return 1 - d/radius(count)


def radius(count):
    return LE * 0.75**count


def train(train_vectors):
    print("Starting to train...")
    EPOCHS = 10
    plt.figure()
    plt.title("S matrix at first")
    s_mat = []
    for i in range(40):
        v = []
        for j in range(40):
            a = random()
            b = random()
            c = random()
            v.append((a, b, c))
            plt.subplot(40, 40, i*40+j+1)
            img = np.zeros([5, 5, 3], np.uint8)
            img = cv2.rectangle(img, (0, 0), (5, 5), (int(a*256), int(b*256), int(c*256)), -1)
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])
        s_mat.append(v)

    for epoch in range(EPOCHS):
        for vec in train_vectors:
            i, j = find_bmu(s_mat, vec)
            update_som(s_mat, i, j, vec, L(ALPHA, epoch), epoch)

    print("Train finished")
    plt.figure()
    plt.title("S matrix at last")
    for i in range(40):
        for j in range(40):
            a, b, c = s_mat[i][j]
            plt.subplot(40, 40, i*40+j+1)
            img = np.zeros([5, 5, 3], np.uint8)
            img = cv2.rectangle(img, (0, 0), (5, 5), (int(a*256), int(b*256), int(c*256)), -1)
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])


if __name__ == "__main__":
    LE = 16
    ALPHA = 0.8
    t_vec = []
    plt.figure()

    for i in range(4):
        a = random()
        b = random()
        c = random()
        t_vec.append([a, b, c])
        plt.subplot(2, 2, i+1)
        img = np.zeros([50, 100, 3], np.uint8)
        img = cv2.rectangle(img, (0, 0), (100, 50), (int(a*256), int(b*256), int(c*256)), -1)
        plt.imshow(img)
        plt.title("train vector number" + str(i+1))
        plt.xticks([]), plt.yticks([])
    train(t_vec)
    plt.show()
