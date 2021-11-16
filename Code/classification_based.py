# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:44:38 2021

@author: Asus
"""

import cv2
import tensorflow as tf
import keras
import os
from glob import glob
import sys
from tqdm import tqdm
import uuid
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, schedules, SGD
from tensorflow.keras.layers import Flatten, Dense, Conv2D, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings

# ignore warnings
warnings.filterwarnings('ignore')

ROOT = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/"
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 32
EPOCH_SIZE = 100
TRAINING = True
PROCESSIMG = False
DATA_BALANCE = False
USE_GPU = True

"""
Data Preparation
"""    
if PROCESSIMG:
    # create dataset folders
    if not os.path.isdir(ROOT+"Datasets"):
        os.mkdir(ROOT+"Datasets")
    
    if not os.path.isdir(ROOT+"Datasets/classification"):
        os.mkdir(ROOT+"Datasets/classification")
        
    if not os.path.isdir(ROOT+"Datasets/classification/train"):
        os.mkdir(ROOT+"Datasets/classification/train")
        
    if not os.path.isdir(ROOT+"Datasets/classification/test"):
        os.mkdir(ROOT+"Datasets/classification/test")
        
    if not os.path.isdir(ROOT+"Datasets/classification/val"):
        os.mkdir(ROOT+"Datasets/classification/val")
    
    # Create new sub folders for all the classes
    class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")

    for i in tqdm(range(len(class_list))):
        count = 0
        if not os.path.isdir(ROOT+"Datasets/classification/test/"+class_list[i]):
            os.mkdir(ROOT+"Datasets/classification/test/"+class_list[i])
            
        if not os.path.isdir(ROOT+"Datasets/classification/train/"+class_list[i]):
            os.mkdir(ROOT+"Datasets/classification/train/"+class_list[i])
            
        if not os.path.isdir(ROOT+"Datasets/classification/val/"+class_list[i]):
            os.mkdir(ROOT+"Datasets/classification/val/"+class_list[i])
                  
        for file in glob(ROOT + "Datasets/classification_data/train_data/" + class_list[i] +"/*.jpg"):
            img = cv2.imread(file)
            
            face_cascade = cv2.CascadeClassifier(ROOT+'opencv/haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5)
            
            if len(faces) == 1:
                count += 1
                path = ROOT + "Datasets/classification/train/" + class_list[i] + "/"
                
                for (x, y, w, h) in faces:
                    crop_img = img[y:y+h, x:x+w]
                    cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), crop_img)
                    
                if (count==20):
                    break
        
        for file in glob(ROOT + "Datasets/classification_data/test_data/" + class_list[i] +"/*.jpg"):
            img = cv2.imread(file)
            path = ROOT + "Datasets/classification/test/" + class_list[i] + "/"
            cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), img)
        
        for file in glob(ROOT + "Datasets/classification_data/val_data/" + class_list[i] +"/*.jpg"):
            img = cv2.imread(file)
            path = ROOT + "Datasets/classification/val/" + class_list[i] + "/"
            cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), img)

if DATA_BALANCE:
    class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")
    print("check data imbalance") 
                
    for i in tqdm(range(len(class_list))):
        numberofimage = os.listdir(ROOT+"Datasets/classification/train/"+class_list[i])
        
        if len(numberofimage) < 20:
            count = len(numberofimage)
            for file in glob(ROOT + "Datasets/classification_data/train_data/" + class_list[i] +"/*.jpg"):
                img = cv2.imread(file)
                count += 1
                path = ROOT + "Datasets/classification/train/" + class_list[i] + "/"
                cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), img)
                    
                if (count==20):
                    break

# training set image data generator
# training set flow from directory
Train_dir = ROOT+"Datasets/classification/train/"
datagen = ImageDataGenerator(validation_split=0.2, rescale = 1./255, rotation_range = 20, height_shift_range = 0.2,
                             zoom_range = 0.1, horizontal_flip = True, shear_range = 0.2, vertical_flip = True,
                             width_shift_range = 0.2, fill_mode = 'nearest')
training_set = datagen.flow_from_directory(Train_dir, target_size = (IMG_WIDTH,IMG_HEIGHT),
                                                 color_mode = "rgb", batch_size = BATCH_SIZE, subset='training', 
                                                 shuffle = True, class_mode ='categorical')

# image data generator for validation images
# validation set using flow from directory
Validation_dir = ROOT+"Datasets/classification/val/"
valid_datagen=ImageDataGenerator(validation_split=0.2,rescale=1./255)
validation_set = valid_datagen.flow_from_directory(Train_dir, target_size = (IMG_WIDTH,IMG_HEIGHT),
                                                   color_mode = "rgb", batch_size = BATCH_SIZE, shuffle = True, subset='validation',
                                                   class_mode = 'categorical')

# image data generator for test images
# test set using flow from directory
Test_dir = ROOT+"Datasets/classification/test/"
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory(Test_dir, target_size = (IMG_WIDTH,IMG_HEIGHT), 
                                            color_mode = "rgb", batch_size = 1, shuffle = False, class_mode = 'categorical')

# get number of classes. its great for flexibility
folders = glob(Train_dir + "/*")

# store the faces into a dictionary to act as a face database.
ResultMap = {}
TrainClasses = training_set.class_indices
for faceValue,faceName in zip(TrainClasses.values(), TrainClasses.keys()):
    ResultMap[faceValue]=faceName

"""
Model Settings
    - CNN model architecture
"""

save_model = ROOT + "Model/cnn.h5"
save_best_model = ROOT + "Model/cnn_best.h5"

pretrianed_model = ResNet50V2(include_top=False, weights="imagenet", pooling="avg", input_shape = (IMG_WIDTH,IMG_HEIGHT,3), classifier_activation='softmax')

pretrianed_model.trainable = False
        
def cnn_model():
    model = Sequential([
        pretrianed_model,
        Flatten(),
        Dense(2048,activation="relu"),
        Dense(4096,activation="relu"),
        Dense(len(folders), activation='softmax')
    ])
    
    # change learning rate according to number of epoch.
    num_train_steps = (len(training_set)//BATCH_SIZE) * EPOCH_SIZE
    lr_scheduler = schedules.PolynomialDecay(
        initial_learning_rate = 1e-2,
        end_learning_rate = 1e-3,
        decay_steps = num_train_steps
    )
    
    # optimizers choices
    opt1 = Adam(learning_rate = 0.001)
    
    # compile model
    model.compile(loss = 'categorical_crossentropy', optimizer = opt1, metrics = ["accuracy"])
    
    # print out model summary
    model.summary()
    
    # early stopping is useful if you want to get the best model
    #early = EarlyStopping(monitor = 'val_loss', patience = 20, verbose = 1, mode = 'min')
    
    # uncomment to enable save the best model according to val_accuracy
    best_model = ModelCheckpoint(save_best_model, monitor = 'val_accuracy', verbose = 1, save_best_only = True, save_weights_only = False)
    
    # model fit
    history = model.fit(training_set
                        ,verbose = 1
                        ,steps_per_epoch = (len(training_set)//BATCH_SIZE)*3
                        ,validation_data = validation_set
                        ,validation_steps= (len(validation_set)//BATCH_SIZE)*3
                        ,epochs = EPOCH_SIZE
                        ,callbacks = [best_model]
                        )
    
    # fine tuning
    pretrianed_model.trainable = True
    model.summary()
    
    # change learning rate according to number of epoch.
    num_train_steps = (len(training_set)//BATCH_SIZE) * (EPOCH_SIZE/2)
    lr_scheduler = schedules.PolynomialDecay(
        initial_learning_rate = 1e-3,
        end_learning_rate = 1e-4,
        decay_steps = num_train_steps
    )
    
    # optimizers choices
    opt1 = Adam(learning_rate = 0.0001)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = opt1, metrics = ["accuracy"])
    
    # model fit
    history = model.fit(training_set
                        ,verbose = 1
                        ,steps_per_epoch = (len(training_set)//BATCH_SIZE)*3
                        ,validation_data = validation_set
                        ,validation_steps= (len(validation_set)//BATCH_SIZE)*3
                        ,epochs = EPOCH_SIZE/2
                        ,callbacks = [best_model]
                        )
    
    # save model
    model.save(save_model)
    
    return model, history

if (TRAINING):
    model, history = cnn_model()

"""
Model evaluation
    - unpack all the loss and accuracy value
    - plot it down using plt
"""
    
def model_evaluation():
    print("Average training accuracy: ", round(np.mean(history.history['accuracy'])*100,2))
    print("Average validation accuracy: ", round(np.mean(history.history['val_accuracy'])*100,2))
    
    accuracy = history.history["accuracy"]
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(accuracy))
    
    fig = plt.figure(figsize=(14,7))
    plt.plot(epochs, accuracy, 'r', label="Training Accuracy")
    plt.plot(epochs, val_accuracy, 'b', label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    
    fig2 = plt.figure(figsize=(14,7))
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')
    
    
    plt.show()

if (TRAINING):
    model_evaluation()
    
imgpath = os.path.sep.join(["classification_data", "test_data", "n000003", "0131_01.jpg"])
test_img = image.load_img(imgpath, target_size=(IMG_WIDTH,IMG_HEIGHT))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis=0)
result = model.predict(test_img,verbose=0)

print('Prediction is: ', ResultMap[np.argmax(result)])
