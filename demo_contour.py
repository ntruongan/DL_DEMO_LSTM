# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 19:46:06 2021

@author: ntruo
"""
#%%
import pytesseract
import shutil
import os
import cv2
import random
try:
 from PIL import Image
except ImportError:
 import Image
 #%%
# image_path_in_colab = r"1.jpg"
# extractedInformation = pytesseract.image_to_string(Image.open(image_path_in_colab)).split("\n")


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
# I = cv2.bitwise_not(im_thresh)

# _,labels,stats,centroid = cv2.connectedComponentsWithStats(I)

# result = np.zeros((I.shape[0],I.shape[1],3),np.uint8)

# for i in range(0,labels.max()+1):
#     mask = cv2.compare(labels,i,cv2.CMP_EQ)

#     ctrs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#     result = cv2.drawContours(result,ctrs,-1,(0xFF,0,0))

# plt.figure()
# plt.imshow(result)  
# i = 4
# mask = cv2.compare(labels,i,cv2.CMP_EQ)
# contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
# contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
contours = rec
# contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
# im2, contours, hierarchy = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
# text detection
def contours_text(orig, img, contours):
    i = 0
    texts = []
    for cnt in contours: 
        i = i + 1
        x, y, w, h = cv2.boundingRect(cnt) 

        # Drawing a rectangle on copied image 
        rect = cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 255, 255),2) 
        
        cv2.imshow('cnt',rect)
        cv2.waitKey()

        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w] 
        cv2.imwrite("cnt%s.png"%str(i),cropped)
        # Apply OCR on the cropped image 
        config = ('-l vie --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config) 
        texts.append(text)
        print(text)
    return texts
        


#%%

im = cv2.imread('TQ.jpg')
print(im.shape)
im = cv2.resize(im,(800,500))
# im_cv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
# kernel = np.ones((5,5),np.uint8)
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
print(im_gray.shape)
im_blur = blur(im_gray)
print(im_blur.shape)
im_thresh = threshold(im_blur) 
# im = cv2.threshold(im_blur, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)  
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
dilation = cv2.dilate(im, rect_kernel, iterations = 1)
# dilation = cv.dilate(img,kernel,iterations = 1)   
# im_thresh = cv2.dilate(im_thresh,kernel,iterations = 1) 
contours = rec
# contours, _ = cv2.findContours(im_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
a = contours_text(orig = im, img = im_thresh, contours = contours)

#%%
import numpy as np
def mydef():
    large = cv2.imread('TQ.jpg')
    # rgb = large
    rgb = cv2.resize(large,(800,500))
    # rgb = cv2.pyrDown(rgb)
    cv2.imshow('down', rgb)
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', small)
    
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    cv2.imshow('grad', grad)
    
    
    
    _, bw = cv2.threshold(grad, 40.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imshow('threshold', bw)
    
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('connected', connected)
    
    
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 8))
    dilation = cv2.dilate(connected, rect_kernel, iterations = 2)
    cv2.imshow('dilation', dilation)
    
    
    contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    
    mask = np.zeros(bw.shape, dtype=np.uint8)
    texts = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
    
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
    
        cropped = small[y:y + h, x:x + w] 
        config = ('-l vie --oem 1 --psm 3')
        text = pytesseract.image_to_string(cropped, config=config) 
        texts.append(text)
        print(text)
    cv2.imshow('rects', rgb)
    return texts
#%%
# cv2.imshow('rects', rgb)
a = mydef()
#%%