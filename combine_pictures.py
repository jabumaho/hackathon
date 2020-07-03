import os
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm

img_size = 256

bad = []
good = []

data = []

for img_name in tqdm(os.listdir('PIM_Dataset/bad_piece')):
    img = cv2.imread(os.path.join('PIM_Dataset/bad_piece', img_name), cv2.IMREAD_GRAYSCALE)
    img = img[250:img.shape[0], 0:img.shape[1]]
    bad.append([img, img_name])


for i in tqdm(range(int(len(bad) / 3))):
    ind = i * 3
    if bad[ind][1].split('_')[1][0] != '1' or bad[ind + 1][1].split('_')[1][0] != '2' or bad[ind + 2][1].split('_')[1][0] != '3':
        continue
    comb_img = np.concatenate((bad[ind][0], bad[ind + 1][0], bad[ind + 2][0]), axis=1)
    comb_img = cv2.resize(comb_img, (img_size, img_size))
    data.append([np.array(comb_img), np.array([0, 1])])

bad.clear()

for img_name in tqdm(os.listdir('PIM_Dataset/good_piece')):
    img = cv2.imread(os.path.join('PIM_Dataset/good_piece', img_name), cv2.IMREAD_GRAYSCALE)
    img = img[250:img.shape[0], 0:img.shape[1]]
    good.append([img, img_name])

for i in tqdm(range(int(len(good) / 3))):
    ind = i * 3
    if good[ind][1].split('_')[1][0] != '1' or good[ind + 1][1].split('_')[1][0] != '2' or good[ind + 2][1].split('_')[1][0] != '3':
        continue
    comb_img = np.concatenate((good[ind][0], good[ind + 1][0], good[ind + 2][0]), axis=1)
    comb_img = cv2.resize(comb_img, (img_size, img_size))
    data.append([np.array(comb_img), np.array([1, 0])])

good.clear()

shuffle(data)

test_data = data[:100]
training_data = data[100:]

np.save('train_data_{}.npy'.format(img_size), training_data)
np.save('test_data_{}.npy'.format(img_size), test_data)
