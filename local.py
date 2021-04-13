# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:29:03 2021

@author: ntruo
"""


import pytesseract
import shutil
import os
import cv2
import random
try:
 from PIL import Image
except ImportError:
 import Image


#%% cv2 part
import cv2
# preprocessing
# gray scale
def gray(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(r"./preprocess/img_gray.png",img)
    return img

# blur
def blur(img) :
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    cv2.imwrite(r"./preprocess/img_blur.png",img)    
    return img_blur

# threshold
def threshold(img):
    #pixels with value below 100 are turned black (0) and those with higher value are turned white (255)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]    
    cv2.imwrite(r"./preprocess/img_threshold.png",img)
    return img

import numpy as np
import matplotlib.pyplot as plt

#%%

#name
rec = []
lines = []
pointA = [335,197]
pointB = [335,150]
pointC = [790,197]
pointD = [790,150]
lines.append(pointA)
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))
# rec = [np.array(rec)]

# birth
lines = []
pointA = [370,200]
pointB = [370,240]
pointC = [790,200]
pointD = [790,240]
lines.append(pointA)
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))


# major
lines = []
pointA = [370,240]
pointB = [370,280]
pointC = [790,240]
pointD = [790,280]
lines.append(pointA)
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))


#  year entry
lines = []
pointA = [400,310]
pointB = [400,350]
pointC = [790,310]
pointD = [790,350]
lines.append(pointA)
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))


# bank
lines = []
pointA = [250,353]
pointB = [250,385]
pointC = [790,353]
pointD = [790,385]
lines.append(pointA)
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))

# id
lines = []
pointA = [25,435]
pointB = [25,490]
pointC = [260,435]
pointD = [260,490]
lines.append(pointA)    
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))


# ava
lines = []
pointA = [52,150]
pointB = [52,380]
pointC = [242,150]
pointD = [242,380]
lines.append(pointA)    
lines.append(pointB)
lines.append(pointC)
lines.append(pointD)
rec.append(np.array(lines))
#%%
import re

def contours_text(orig, img, contours):
    i = 0
    dict_info = {}
    countours_label = ["Name", "DOB", "Faculty","Admission Year", "Bank ID","Student ID","Image"]
    texts = []
    for idx, cnt in enumerate(contours): 
        i = i + 1
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255),2) 
        
        cv2.imshow('cnt',rect)
        cv2.waitKey()

        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w] 
        # cropped = img[y:y + h, x:x + w] 
        cv2.imwrite("cnt%s.png"%str(i),cropped)
        # Apply OCR on the cropped image 

        # print(text)
        if countours_label[idx] == "Image":
            dict_info[countours_label[idx]] = cropped
            cv2.imshow('avt',cropped)
        else:
            config = ('-l vie --oem 1 --psm 3')
            text = pytesseract.image_to_string(cropped, config=config) 
            text = re.sub(r"[\n\x0c]","",text).strip()
            dict_info[countours_label[idx]] = text
            print("%s: %s" % (countours_label[idx], text))
        texts.append(text)
        
    return dict_info
        

#%%
im = cv2.imread('2.jpg')
im = cv2.resize(im,(800,500))
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_blur = blur(im_gray)
im_thresh = threshold(im_blur) 
contours = rec
a = contours_text(orig = im, img = im_thresh, contours = contours)