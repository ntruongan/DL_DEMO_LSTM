# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 14:04:20 2021

@author: ntruo
"""

import jsonpickle
import tensorflow as tf
import flask
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pytesseract
import shutil
import os
import cv2
import io
import random
import base64
try:
 from PIL import Image
except ImportError:
 import Image
from flask import Flask, request, Response
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)


def build_contours():
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
    return rec

def img_package(img):
    img = Image.fromarray(img.astype("uint8"))
    rawBytes = io.BytesIO()
    img.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return img_base64

def model_predict(orig, img, contours):
    i = 0
    dict_info = {}
    dict_image = {}
    countours_label = ["Name", "DOB", "Faculty","Admission Year", "Bank ID","Student ID","Image"]
    # texts = []
    for idx, cnt in enumerate(contours): 
        i = i + 1
        x, y, w, h = cv2.boundingRect(cnt) 


        # Cropping the text block for giving input to OCR 
        cropped = orig[y:y + h, x:x + w] 
        # cropped = img[y:y + h, x:x + w] 
        # cv2.imwrite("cnt%s.png"%str(i),cropped)
        # Apply OCR on the cropped image 

        # print(text)
        if countours_label[idx] == "Image":
            img_str = img_package(cropped)
            dict_image[countours_label[idx]] = img_str
            # dict_info1 = cropped
            # cv2.imshow('avt',cropped)
        else:
            config = ('-l vie --oem 1 --psm 3')
            text = pytesseract.image_to_string(cropped, config=config) 
            print(text)
            text = re.sub(r"[\n\x0c:~.?<>!#$%^&*()+=_|]","",text).strip()
            dict_info[countours_label[idx]] = text
            print("%s: %s" % (countours_label[idx], text))
        # texts.append(text)
        # 
    return dict_info, dict_image


def gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# blur
def blur(img) :
    img_blur = cv2.GaussianBlur(img,(5,5),0)
    return img_blur

# threshold
def threshold(img):
    img = cv2.threshold(img, 100, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1] 
    return img

def img_preprocessing(img):

    img = gray(img) 
    print(img.shape) 
    img = blur(img)
    img = img.astype("uint8") 
    img = threshold(img)
    return img

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
    data = {"success": False}
    # preds = {"success": False}
    # r = request
    # convert string of image data to uint8
    # img = np.fromstring(r.data, np.uint8)
    # decode image
    if flask.request.method == "POST":
        if flask.request.files.get("input_"):
            input_ = flask.request.files["input_"]
            # print((input_))
            # input_ = Image.open(input_.stream)
            # input_ = np.array(input_) 
            
            # Convert RGB to BGR 
            # input_ = input_[:, :, ::-1].copy() 
            # input_ = np.fromstring(input_, np.uint8)
            # print(input_.shape)
            # input_ = cv2.imdecode(input_, cv2.IMREAD_COLOR) # cv2.IMREAD_COLOR in OpenCV 3.1
            # input_ = np.array(input_)
            input_ = np.fromstring(input_.read(), np.uint8)
            input_ = cv2.imdecode(input_,cv2.IMREAD_COLOR)
            print(input_.shape)
            img = input_.astype("uint8") 
            img = cv2.resize(img, (800,500))
            input_prepro = img_preprocessing(img)
            preds, image_pred = model_predict(img, input_prepro, build_contours())
            # pred_word = tokenizer.index_word[preds]
            data["predictions"] = preds
            data["success"] = True
            # data["image"] = image_pred
    # return flask.jsonify(preds)
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run(debug=False)