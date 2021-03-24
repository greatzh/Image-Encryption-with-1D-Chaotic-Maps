from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
from operator import mod
import numpy as np
import random
import math
import cv2
import time
from math import log
# P 256*256 double
def encrypt(P, K):
    r1, r2, r3, r4, r5 = K[0], K[1], K[3], K[4], K[5]
    size = P.shape
    w = size[0]
    l = size[1]
    h = 1
    range_i = 2 * w + 2 * l
    X0 = np.zeros(1024)
    X = np.zeros(257)
    C = np.zeros(257)
    C1 = np.zeros((257, 257))
    X0[0] = K[2]
    # LTS system Logistic-Tent System
    for i in range(1,range_i):
        X0[i] = mod(r1 * (1 - X0[i - 1]) * X0[i - 1] + 1 * (4 - r1) / 4 * (X0[i - 1] / 0.5 * float(X0[i - 1] < 0.5) + (1 - X0[i - 1]) / (1 - 0.5) * float(X0[i - 1] >= 0.5)), 1)
    # Random Pixels Generate
    print(X0.size)
    C0 = np.random.randint(256, size = range_i) - 1

    # 1st time of 4
    for j in range(w):
        # 1. inserts a random pixel in each row of the raw image
        X[0] = mod(j * X0[j], 1)
        C[0] = C0[j]
        # 3. 1D Sbstitution
        for i in range(l):
            X[i + 1] = mod(r2 * (1 - X[i]) * X[i] + 1 * (4 - r2) / 4 * (X[i] / 0.5 * float(X[i] < 5) + (1 - X[i]) / (1 - 0.5) * float(X[i] >= 0.5)), 1)
            tmp = math.floor(mod(X[i] * 10 ** 10, 256))
            C[i + 1] = (C[i].astype(dtype = np.int32) | tmp) | P[j, i].astype(dtype = np.int32)
        # 5. Image Rotation
        tmp_arry = C[1: l + 1].copy()
        P[j, :] = tmp_arry

    # 2nd time of 4
    for i in range(l):
        X[0] = mod(i * X0[i + w], 1)
        C1[0,0] = C0[i + w]
        for j in range(w):
            X[j + 1] = mod(r3 * (1 - X[j]) * X[j] + 1 * (4 - r3) / 4 * (X[j] / 0.5 * float(X[j] < 5) + (1 - X[j]) / (1 - 0.5) * float(X[j] >= 0.5)), 1)
            C1[j + 1, 0] = (C1[j,0].astype(dtype = np.int32) | math.floor(mod(X[j] * 10 ** 10, 256))) | P[j, i].astype(dtype = np.int32)
        P[:, i] = C1[1: w + 1, 0]

    # 3rd time of 4
    for j in range(w):
        X[0] = mod(j * X0 [j + w + l], 1)
        C2 = np.zeros(257)
        C2[1] = C0[j + w + l]
        for i in range(l, 0, -1):
            X[l - i + 1] = mod(r4 * (1 - X[l - i + 0]) * X[l - i + 0] + 1 * (4 - r4) / 4 * (X[l - i + 0] / 0.5 * float(X[l - i + 0] < 5) + (1 - X[l - i + 0]) / (1 - 0.5) * float(X[l - i + 0] >= 0.5)), 1)
            tmp2 = C2[i].astype(dtype = np.int32)
            tmp3 = math.floor(mod(X[l - i + 1] * 10 ** 10, 256))
            tmp4 = P[j, i-1].astype(dtype = np.int32)
            C2[i - 1] = (tmp2 | tmp3) | tmp4
        P[j, :] = C2[0: l]
    
    # 4th time of 4
    for i in range(l):
        X = np.zeros(257)
        X[0] = mod(i * X0[i + 2 * w + l], 1)
        C3 = np.zeros((257, 1))
        C3[w, 0] = C0[i + 2 * w + l]
        for j in range(w, 0, -1):
            X[w - j + 1] = mod(r5 * (1 - X[w - j + 0]) * X[w - j + 0] + 1 * (4 - r5) / 4 * (X[w - j + 0] / 0.5 * float(X[w - j + 0] < 5) + (1 - X[w - j + 0]) / (1 - 0.5) * float(X[w - j + 0] >= 0.5)), 1)
            C2[j - 1] = (C2[j].astype(dtype = np.int32) | math.floor(mod(X[w - j + 1] * 10 ** 10, 256))) | P[j - 1, i].astype(dtype = np.int32)
        P[:, i] = C3[0: w, 0]
    E = P.copy()
    return E

def decrypt(P, K):
    r1, r2, r3, r4, r5 = K[0], K[1], K[3], K[4], K[5]
    size = P.shape
    w = size[0]
    l = size[1]
    h = 1
    range_i = 2 * w + 2 * l
    X0 = np.zeros(1024)
    X = np.zeros(257)
    C = np.zeros(257)
    C1 = np.zeros((257, 257))
    C2 = np.zeros(257)
    X0[0] = K[2]
    for i in range(range_i):
        X0[i] = mod(r1 * (1 - X0[i - 1]) * X0[i - 1] + 1 * (4 - r1) / 4 * (X0[i - 1] / 0.5 * float(X0[i - 1] < 5) + (1 - X0[i - 1]) / (1 - 0.5) * float(X0[i - 1] >= 0.5)), 1)
  
    P1 = np.zeros((w + 1, l))
    P1[0 : w, 0 : l] = P
    for i in range(l):
        X[0] = mod(i * X0[i + 2 * w + l], 1)
        C3 = np.zeros((w + 1, l))
        for j in range(w):
            # inverse 1D Substitution
            X[j + 1] = mod(r5 * (1 - X[j]) * X[j] + 1 * (4 - r5) / 4 * (X[j] / 0.5 * float(X[j] < 5) + (1 - X[j]) / (1 - 0.5) * float(X[j] >= 0.5)), 1)
            tmp1 = P1[j, i].astype(dtype = np.int32)
            tmp2 = math.floor(mod(X[w + 1 - j - 2] * 10 ** 10, 256))
            tmp3 = P1[j + 1, i].astype(dtype = np.int32)
            C3[j, 0] = (tmp1 | tmp2) | tmp3
        P[:, i] = C3[0 : w, 0]

    P2 = np.zeros((w, l + 1))
    P2[0 : w, 0 : l] = P
    for j in range(w):
        X[0] = mod(j * X0[j + w + l], 1)
        for i in range(l):
            X[i + 1] = mod(r4 * (1 - X[i]) * X[i] + 1 * (4 - r4) / 4 * (X[i] / 0.5 * float(X[i] < 5) + (1 - X[i]) / (1 - 0.5) * float(X[i] >= 0.5)), 1)
            C2[i] = (P2[j, i].astype(dtype = np.int32) | math.floor(mod(X[l + 1 - i - 2] * 10 ** 10, 256))) | P2[j, i + 1].astype(dtype = np.int32)
        P[j, :] = C2[0 : l]

    P3 = np.zeros((w + 1, l))
    P3[1 : w + 1, 0 : l] = P
    for i in range(l):
        X[0] = mod(i * X0[i + w], 1)
        C1[0, 0] = 0
        for j in range(w):
            X[j + 1] = mod(r3 * (1 - X[j]) * X[j] + 1 * (4 - r3) / 4 * (X[j] / 0.5 * float(X[j] < 5) + (1 - X[j]) / (1 - 0.5) * float(X[j] >= 0.5)), 1)
        for j in range(w, 0, -1):
            t1 = P3[j + 0, i].astype(dtype = np.int32)
            t2 = math.floor(mod(X[j] * 10 ** 10, 256))
            t3 = P3[j - 1, i].astype(dtype = np.int32)
            C1[j, 0] = (t1 | t2) | t3
        # print(C1.size)
        # print('2',P.size)
        P[:, i] = C1[0: w, 1]
   
    P4 = np.zeros((w, l + 1))
    P4[0 : w, 1 : l + 1] = P
    for j in range(w):
        X[0] = mod(j * X0[j], 1)
        for i in range(l):
            X[i + 1] = mod(r2 * (1 - X[i]) * X[i] + 1 * (4 - r2) / 4 * (X[i] / 0.5 * float(X[i] < 5) + (1 - X[i]) / (1 - 0.5) * float(X[i] >= 0.5)), 1)
        for i in range(l, 1, -1):
            tmp4 = P4[j, i + 0].astype(dtype = np.int32)
            tmp5 = math.floor(mod(X[i] * 10 ** 10, 256))
            tmp6 = P4[j, i - 1].astype(dtype = np.int32)
            C[i] = (tmp4 | tmp5) | tmp6
        P[j - 1,:] = C[0:l]
    P = P[1: w - 1, 1 : l]
    D = P.copy()
    return D
# get the entropy
def entropy(props, base=2):
    sum = 0
    for prop in props:
        sum += prop * log(prop, base)
    return sum * -1
 
if __name__ == '__main__':
    K = [3.99, 3.96, 0.6, 4, 3.999, 3.997]
    img = plt.imread('./misc/5.1.12.tiff')
    img = np.array(img)
    P = np.array(img)

    tic = time.time()
    C = encrypt(img, K)
    
    toc = time.time()
    D = decrypt(C, K)

    fig = plt.figure()
    #plt.figure(figsize=(10,5)) # set the windows size
    plt.suptitle('Final Result') # image title
    plt.subplot(2,2,1), plt.title('Plaintext Image')
    plt.imshow(P,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('Ciphertext Image')
    plt.imshow(C,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('Decrypted Image')
    plt.imshow(D,cmap='gray'), plt.axis('on')
    plt.show() 
    
    # Histogram 
    plt.figure('cat')
    arr1 = P.flatten()
    arr2 = C.flatten()
    arr3 = D.flatten()
    plt.figure('Histogram')
    plt.subplot(221)
    n, bins, patches = plt.hist(arr1)
    plt.subplot(222)
    n, bins, patches = plt.hist(arr2)
    plt.subplot(223)
    n, bins, patches = plt.hist(arr3)
    plt.show()
    
    # entropy
    props = []
    for i in C:
        props.append([i, 1-i])    
    y = [entropy(i) for i in props]
    plt.plot(C,y)
    plt.xlabel("p(x)")
    plt.ylabel("H(x)")
    plt.show()

    misc.imsave('fruiten.png', C)

    D3 = encrypt(P, K)

    fig = plt.figure()
    plt.subplot(2,2,1), plt.title('P')
    plt.imshow(P,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('Encrypting')
    plt.imshow(C,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('Encrypting')
    plt.imshow(D3,cmap='gray'), plt.axis('on')
    plt.show() 
    
    DK1 = decrypt(D3, K)
    K1 = K.copy()
    K1[2] = 0.2
    DK2 = decrypt(D3, K1)

    fig = plt.figure()
    plt.subplot(2,2,1), plt.title('Encrypting P')
    plt.imshow(C,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('Decrypting')
    plt.imshow(DK1,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('Decrypting')
    plt.imshow(DK2,cmap='gray'), plt.axis('on')
    plt.show() 
