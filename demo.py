from PIL import Image
import matplotlib.pyplot as plt
from scipy import misc
from operator import mod
import numpy as np
import random
import math
import cv2
import time
import sys
from math import log
# P 256*256 double
def encrypt(P, K):
    r1, r2, r3, r4, r5 = K[0], K[1], K[3], K[4], K[5]
    size = P.shape
    if(len(size) == 2):
        h = 1
    else:
        h = size[2]
    w = size[0]
    l = size[1]
    range_i = 2 * w + 2 * l
    X0 = np.zeros(range_i)
    X = np.zeros(w + 1)
    C = np.zeros(w + 1)
    C1 = np.zeros((w + 1, l + 1))
    X0[0] = K[2]
    # LTS system Logistic-Tent System
    for i in range(1,range_i):
        X0[i] = mod(r1 * (1 - X0[i - 1]) * X0[i - 1] + 1 * (4 - r1) / 4 * (X0[i - 1] / 0.5 * float(X0[i - 1] < 0.5) + (1 - X0[i - 1]) / (1 - 0.5) * float(X0[i - 1] >= 0.5)), 1)
    # Random Pixels Generate
    # flag of 
    C0 = np.random.randint(256, size = range_i) - 1

    # 1st time of 4: right
    for j in range(w):
        # 1. inserts a random pixel in each row of the raw image
        X[0] = mod(j * X0[j], 1)
        C[0] = C0[j]
        # 3. 1D Sbstitution
        for i in range(l):
            X[i + 1] = mod(r2 * (1 - X[i]) * X[i] + (4 - r2) / 2 * (X[i] * float(X[i] < 0.5) + (1 - X[i]) * float(X[i] >= 0.5)), 1)
            C[i + 1] = (C[i].astype(dtype = np.int32) ^ math.floor(mod(X[i] * 10 ** 10, 256))) ^ P[j, i].astype(dtype = np.int32)
        # 5. Image Rotation
        P[j, :] = C[1: l + 1].copy()
    
    # 2nd time of 4: down
    for i in range(l):
        X[0] = mod(i * X0[i + w], 1)
        C1[0,0] = C0[i + w]
        for j in range(w):
            X[j + 1] = mod(r3 * (1 - X[j]) * X[j] + 1 * (4 - r3) / 2 * (X[i] * float(X[i] < 0.5) + (1 - X[i]) * float(X[i] >= 0.5)), 1)
            C1[j + 1, 0] = (C1[j,0].astype(dtype = np.int32) ^ math.floor(mod(X[j] * 10 ** 10, 256))) ^ P[j, i].astype(dtype = np.int32)
        P[:, i] = C1[1: w + 1, 0].copy()
    
    # 3rd time of 4 left
    for j in range(w):
        X[0] = mod(j * X0 [j + w + l], 1)
        C2 = np.zeros(l + 1)
        C2[1] = C0[j + w + l]
        for i in range(l, 0, -1):
            X[l - i + 1] = mod(r4 * (1 - X[l - i + 0]) * X[l - i + 0] + 1 * (4 - r4) / 2 * (X[l - i + 0] * float(X[l - i + 0] < 0.5) + (1 - X[l - i + 0]) * float(X[l - i + 0] >= 0.5)), 1)
            tmp2 = C2[i].astype(dtype = np.int32)
            tmp3 = math.floor(mod(X[l - i + 0] * 10 ** 10, 256))
            tmp4 = P[j, i - 1].astype(dtype = np.int32)
            C2[i - 1] = (tmp2 ^ tmp3) ^ tmp4
        P[j, :] = C2[0: l].copy()
   
    # 4th time of 4 up
    for i in range(l):
        X = np.zeros(w + 1)
        X[0] = mod(i * X0[i + 2 * w + l], 1)
        C3 = np.zeros((w + 1, 1))
        C3[w, 0] = C0[i + 2 * w + l]
        for j in range(w, 0, -1):
            X[w - j + 1] = mod(r5 * (1 - X[w - j + 0]) * X[w - j + 0] + (4 - r5) / 2 * (X[w - j + 0] * float(X[w - j + 0] < 0.5) + (1 - X[w - j + 0]) * float(X[w - j + 0] >= 0.5)), 1)
            C3[j - 1] = (C3[j].astype(dtype = np.int32) ^ math.floor(mod(X[w - j + 0] * 10 ** 10, 256))) ^ P[j - 1, i].astype(dtype = np.int32)
        P[:, i] = C3[0: w, 0].copy()
        
    E = P.copy()
    return E

def decrypt(P, K):
    r1, r2, r3, r4, r5 = K[0], K[1], K[3], K[4], K[5]
    size = P.shape
    if(len(size) == 2):
        h = 1
    else:
        h = size[2]
    w = size[0]
    l = size[1]
    range_i = 2 * w + 2 * l
    X0 = np.zeros(range_i)
    X = np.zeros(w + 1)
    C = np.zeros(w + 1)
    C1 = np.zeros((w + 1, l + 1))
    C2 = np.zeros(l + 1)
    X0[0] = K[2]
    for i in range(1,range_i):
        X0[i] = mod(r1 * (1 - X0[i - 1]) * X0[i - 1] + 1 * (4 - r1) / 4 * (X0[i - 1] / 0.5 * float(X0[i - 1] < 0.5) + (1 - X0[i - 1]) / (1 - 0.5) * float(X0[i - 1] >= 0.5)), 1)
    
    # 1st UP
    P1 = np.zeros((w + 1, l))
    P1[0 : w, 0 : l] = P.copy()
    for i in range(l):
        X[0] = mod(i * X0[i + 2 * w + l], 1)
        C3 = np.zeros((w + 1, 1))
        for j in range(w):
            # inverse 1D Substitution
            X[j + 1] = mod(r5 * (1 - X[j]) * X[j] + 1 * (4 - r5) / 4 * (X[j] / 0.5 * float(X[j] < 0.5) + (1 - X[j]) / (1 - 0.5) * float(X[j] >= 0.5)), 1)
        for j in range(w):
            tmp1 = P1[j, i].astype(dtype = np.int32)
            tmp2 = math.floor(mod(X[w + 1 - j - 2] * 10 ** 10, 256))
            tmp3 = P1[j + 1, i].astype(dtype = np.int32)
            C3[j, 0] = (tmp1 ^ tmp2) ^ tmp3
        P[:, i] = C3[0 : w, 0].copy()
    
    # 2nd LEFT
    P2 = np.zeros((w, l + 1))
    P2[0 : w, 0 : l] = P.copy()
    for j in range(w):
        X[0] = mod(j * X0[j + w + l], 1)
        for i in range(l):
            X[i + 1] = mod(r4 * (1 - X[i]) * X[i] + 1 * (4 - r4) / 4 * (X[i] / 0.5 * float(X[i] < 0.5) + (1 - X[i]) / (1 - 0.5) * float(X[i] >= 0.5)), 1)
        for i in range(l):
            C2[i] = (P2[j, i].astype(dtype = np.int32) ^ math.floor(mod(X[l + 1 - i - 2] * 10 ** 10, 256))) ^ P2[j, i + 1].astype(dtype = np.int32)
        P[j, :] = C2[0 : l].copy()
    

    # 3rd DOWN
    P3 = np.zeros((w + 1, l))
    P3[1 : w + 1, 0 : l] = P.copy()
    for i in range(l):
        X[0] = mod(i * X0[i + w], 1)
        C1[0, 0] = 0
        for j in range(w):
            X[j + 1] = mod(r3 * (1 - X[j]) * X[j] + 1 * (4 - r3) / 4 * (X[j] / 0.5 * float(X[j] < 0.5) + (1 - X[j]) / (1 - 0.5) * float(X[j] >= 0.5)), 1)
        for j in range(w, 0, -1):
            t1 = P3[j + 0, i].astype(dtype = np.int32)
            t2 = math.floor(mod(X[j - 1] * 10 ** 10, 256))
            t3 = P3[j - 1, i].astype(dtype = np.int32)
            C1[j - 1, 0] = (t1 ^ t2) ^ t3
        # print(C1.size)
        # print('2',P.size)
        P[:, i] = C1[0: w, 0].copy()

    # 4th RIGHT
    P4 = np.zeros((w, l + 1))
    P4[0 : w, 1 : l + 1] = P.copy()
    for j in range(w):
        X[0] = mod(j * X0[j], 1)
        for i in range(l):
            X[i + 1] = mod(r2 * (1 - X[i]) * X[i] + 1 * (4 - r2) / 4 * (X[i] / 0.5 * float(X[i] < 0.5) + (1 - X[i]) / (1 - 0.5) * float(X[i] >= 0.5)), 1)
        for i in range(l, 0, -1):
            tmp4 = P4[j, i + 0].astype(dtype = np.int32)
            tmp5 = math.floor(mod(X[i - 1] * 10 ** 10, 256))
            tmp6 = P4[j, i - 1].astype(dtype = np.int32)
            C[i - 1] = (tmp4 ^ tmp5) ^ tmp6
        P[j,:] = C[0:l].copy()
    D = P[1: w - 1, 1 : l].copy()
    return D
# get the entropy
def entropy(props, base=2):
    sum = 0
    for prop in props:
        sum += prop * log(prop, base)
    return sum * -1
 
if __name__ == '__main__':
    K = [3.99, 3.96, 0.6, 4, 3.999, 3.997]
    img = plt.imread('4.1.01.tiff')
    img = np.array(img)
    P = np.array(img)
    P1 = P.copy()

    tic = time.time()
    C = encrypt(img, K)
    C_temp = C.copy()
    toc = time.time()
    D = decrypt(C_temp, K)

    print('Encrypted time consumed is ', toc - tic)

    plt.figure() # set the windows size
    plt.suptitle('Figure1: Final Result') # image title
    plt.subplot(2,2,1), plt.title('Plaintext Image')
    plt.imshow(P,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('Ciphertext Image')
    plt.imshow(C,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('Decrypted Image')
    plt.imshow(D,cmap='gray'), plt.axis('on')
    plt.show() 
    
    # Histogram 


    arr1 = P.flatten()
    arr2 = C.flatten()
    arr3 = D.flatten()
    plt.figure()
    plt.suptitle('Figure2: Histogram')
    plt.subplot(221), plt.title('Plaintext Image')
    plt.hist(arr1, bins=256) 
    plt.subplot(222), plt.title('Encrypted Image')
    plt.hist(arr2, bins=256)
    plt.subplot(223), plt.title('Decrypted Image')
    plt.hist(arr3, bins=256)
    plt.show()
    
    # entropy
    # props = []
    # for i in C:
    #     props.append([i, 1-i])    
    # y = [entropy(i) for i in props]
    # plt.plot(C,y)
    # plt.xlabel("p(x)")
    # plt.ylabel("H(x)")
    # plt.show()

    # misc.imsave('fruiten.png', C)
    # plaintext sensitivity
    P2 = P.copy()
    P2[128,128] = 0
    C2 = encrypt(P2, K)

    plt.figure()
    plt.suptitle('Figure3: Plaintext Sensitivity')
    plt.subplot(2,2,1), plt.title('P')
    plt.imshow(P,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('P2')
    plt.imshow(P2,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title(' ')
    plt.imshow(C2,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,1), plt.title('C')
    plt.imshow(C,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('C2')
    plt.imshow(C2,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,3), plt.title(' ')
    plt.imshow(C2,cmap='gray'), plt.axis('on')
    plt.show() 
    
    # Key sensitivity
    P11 = P1.copy()
    C3 = encrypt(P1, K)

    plt.figure()
    plt.suptitle('Figure4: Key Sensitivity')
    plt.subplot(2,2,1), plt.title('P')
    plt.imshow(P11,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('C encrypting P with K1')
    plt.imshow(C,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('C3 encrypting P with K2')
    plt.imshow(C3,cmap='gray'), plt.axis('on')
    plt.show() 

    C3_temp = C3.copy()
    C3_temp2 = C3.copy()
    DK1 = decrypt(C3_temp, K)
    K1 = K.copy()
    K1[3] = 0.2
    DK2 = decrypt(C3_temp2, K1)
    plt.figure()
    plt.suptitle('Figure5: Key Sensitivity')
    plt.subplot(2,2,1), plt.title('Encrypting C with K1')
    plt.imshow(C,cmap='gray'), plt.axis('on')
    plt.subplot(2,2,2), plt.title('DK1 decrypting with K1')
    plt.imshow(DK1,cmap='gray'), plt.axis('on') # show as the gray level cmap
    plt.subplot(2,2,3), plt.title('DK2 decrypting with K2')
    plt.imshow(DK2,cmap='gray'), plt.axis('on')
    plt.show()




