import matplotlib.pyplot as plt
from PIL import Image
#img = Image.open('data/stinkbug.png')
import numpy as np
import copy
import os
import cv2

def gen_img_data(args={}):
    return load_girl_data(args)


def save_image(image,addr,num):
    cv2.imwrite(addr+str(num)+".jpg", image)


def load_girl_data(args={}):
    names = ["img_"+str(i)+'.jpg' for i in range(1,101)]
    res = []
    for name in names:        
        rpath = r"images/"
        if not os.path.isfile(rpath+name):
            continue
        img = Image.open(rpath+name)    
        img = np.array(img)
        cat = np.mean(img, axis=2)
        ct = cat.T        
        res.append(ct)
    print('images loaded')
    return np.stack(res)


def imgsshow(compose):
    for c in compose:
        plt.imshow(c)
        plt.axis('off')
        plt.show()

def femnist_save_top_eigen(filename, U, num=1):
    ns = min(num, len(U[0,:]))
    for i in range(ns):
        c = np.reshape(U[:,i], (28,28))
        plt.imshow(c)
        plt.savefig(filename+str(i)+'.png')

