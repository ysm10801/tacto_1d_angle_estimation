

from PIL import Image
import cv2

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import pandas as pd
import glob

import demo_pybullet_digit_PosEst

import random


# print(file_list)


# label_list=[]
# for i in range (datanum):
#     label_name = file_list[i].split('_')[1]
#     label_name = label_name[:label_name.find('.png')]
#     print(label_name)
#     label_list.append([float(label_name)])

# print(label_list)
# print("Number of Data :",len(label_list))

# ex_img_path = glob.glob(PATH_US + '0001' + '*')
# ex_img = img.imread(ex_img_path[0])
# print(ex_img.shape)
# plt.imshow(ex_img)
# plt.show()


size = (60, 80)

def img_resizer():
    path_us = '/home/yang/tacto/examples/data1/'
    
    file_list = glob.glob(path_us + '*')
    file_list.sort()

    file_list_train_raw = random.sample(file_list, 24000)
    file_list_train_raw.sort()
    file_list_val = random.sample(file_list_train_raw, 6000)
    file_list_val.sort()
    file_list_train = [x for x in file_list_train_raw if x not in file_list_val]
    file_list_test = [x for x in file_list if x not in file_list_train_raw]

    img_save(file_list_train, 'train/', size)
    img_save(file_list_val, 'val/', size)
    img_save(file_list_test, 'test/', size)

def img_save(datas, pth, size):
    print("size of {} data : {}".format(pth, len(datas)))
    for i in range(len(datas)):
        file_name = datas[i].split('/')[-1]
        name = '/home/yang/tacto/examples/data_res1/' + pth + file_name
        img = cv2.imread(datas[i])
        img_resize = cv2.resize(img, size)
        cv2.imwrite(name, img_resize)

img_resizer()