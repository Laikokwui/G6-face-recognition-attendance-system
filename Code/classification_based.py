# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 20:44:38 2021

@author: Asus
"""

import cv2
import keras
import os
from glob import glob
import sys
from tqdm import tqdm
import uuid
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
from itertools import cycle

import warnings

import utilities

# ignore warnings
warnings.filterwarnings('ignore')

ROOT = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/"
IMG_WIDTH = 64
IMG_HEIGHT = 64
BATCH_SIZE = 64
EPOCH_SIZE = 50

TRAINING = False
PROCESSIMG = False
DATA_BALANCE = False
USE_GPU = True
NO_IMG_PER_CLASS = 25

"""
Data Preparation
"""
if PROCESSIMG:  
    # Set parameters for yoloV4
    labelsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/obj.names"
    weightsPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj_last.weights"
    configPath = "C:/Users/Asus/Documents/G6-face-recognition-attendance-system/yolov4/yolov4-obj.cfg"
    net, ln, labels = utilities.Yolov4Setup(labelsPath, weightsPath, configPath, True)
    
    # create dataset folders
    if not os.path.isdir(ROOT+"Datasets"):
        os.mkdir(ROOT+"Datasets")
    
    # create a folder to store all images use for this training session
    if not os.path.isdir(ROOT+"Datasets/classification"):
        os.mkdir(ROOT+"Datasets/classification")
    
    # create train folder to store training image
    if not os.path.isdir(ROOT+"Datasets/classification/train"):
        os.mkdir(ROOT+"Datasets/classification/train")
    
    # create test folder to store test images
    if not os.path.isdir(ROOT+"Datasets/classification/test"):
        os.mkdir(ROOT+"Datasets/classification/test")
    
    # create validation folder for validation images
    if not os.path.isdir(ROOT+"Datasets/classification/val"):
        os.mkdir(ROOT+"Datasets/classification/val")
    
    # Create new sub folders for all the classes
    class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")
    
    # loop through all the classes
    for class_name in tqdm(range(len(class_list))):
        count = 0
        if not os.path.isdir(ROOT+"Datasets/classification/test/"+class_list[class_name]):
            os.mkdir(ROOT+"Datasets/classification/test/"+class_list[class_name])
            
        if not os.path.isdir(ROOT+"Datasets/classification/train/"+class_list[class_name]):
            os.mkdir(ROOT+"Datasets/classification/train/"+class_list[class_name])
            
        if not os.path.isdir(ROOT+"Datasets/classification/val/"+class_list[class_name]):
            os.mkdir(ROOT+"Datasets/classification/val/"+class_list[class_name])
        
        # filter image inside original dataset
        for file in glob(ROOT + "Datasets/classification_data/train_data/" + class_list[class_name] +"/*.jpg"):
            # read images
            img = cv2.imread(file, 1)
            
            # detect face location and how many are there
            results = utilities.detect_face(img, net, ln, 0.3, 0.3, objIdx=labels.index("face"))
            
            # if there are only one face in the image
            if len(results) == 1:
                # loop through results to get the face location
                for (i, (prob, bbox, centroid)) in enumerate(results):
                    # get the coordinate from bbox
                    (startX, startY, endX, endY) = bbox
                    
                    # crop the image
                    crop_img = img[startY:endY, startX:endX]
                    
                    # try save the image if not pass
                    try:
                        path = ROOT + "Datasets/classification/train/" + class_list[class_name] + "/"
                        cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), crop_img)
                        count += 1
                    except:
                        pass
            # if the count meet the number of images per class       
            if (count==NO_IMG_PER_CLASS):
                # break the loop
                break
        
        # get test images from test data folder
        for file in glob(ROOT + "Datasets/classification_data/test_data/" + class_list[class_name] +"/*.jpg"):
            img = cv2.imread(file)
            path = ROOT + "Datasets/classification/test/" + class_list[class_name] + "/"
            cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), img)
        
        # get validation images from val data folder
        for file in glob(ROOT + "Datasets/classification_data/val_data/" + class_list[class_name] +"/*.jpg"):
            img = cv2.imread(file)
            path = ROOT + "Datasets/classification/val/" + class_list[class_name] + "/"
            cv2.imwrite(os.path.join(path , '{}.jpg'.format(uuid.uuid1())), img)

# to check whether the amount of data for each folder is the same
if DATA_BALANCE:
    class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")
    print("check data imbalance") 
                
    for i in tqdm(range(len(class_list))):
        numberofimage = os.listdir(ROOT+"Datasets/classification/train/"+class_list[i])
        
        if len(numberofimage) < NO_IMG_PER_CLASS:
            print("Class ", class_list[i], " only have ", len(numberofimage), " of images.")

# training set image data generator
# training set flow from directory
Train_dir = ROOT+"Datasets/classification/train/"
datagen = ImageDataGenerator(validation_split=0.2,rescale = 1./255, rotation_range = 60, height_shift_range = 0.2,
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

"""
Model Settings
    - CNN model architecture
"""

save_model = ROOT + "Model/cnn.h5"
save_best_model = ROOT + "Model/cnn_best.h5"
        
def cnn_model():
    pretrained_model = ResNet152V2(include_top=False ,weights="imagenet" ,pooling=None ,input_shape = (IMG_WIDTH,IMG_HEIGHT,3) ,classifier_activation='softmax')

    pretrained_model.trainable = False

    pretrained_model.summary()
    
    model = Sequential([
        pretrained_model,
        Flatten(),
        Dense(128, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),activation="relu"),
        Dropout(0.5),
        BatchNormalization(),
        Dense(len(folders), activation='softmax')
    ])
    
    METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'), 
            keras.metrics.CategoricalAccuracy(name='c_accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
            keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
    
    # optimizers choices
    opt1 = SGD(learning_rate = 0.1, decay=0.0001)
    
    # compile model
    model.compile(loss = 'categorical_crossentropy', optimizer = opt1, metrics = [METRICS,"accuracy"])
    
    # print out model summary
    model.summary()
    
    # early stopping is useful if you want to get the best model
    early = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1, mode = 'min')
    
    # uncomment to enable save the best model according to val_accuracy
    best_model = ModelCheckpoint(save_best_model, monitor = 'val_accuracy', verbose = 1, save_best_only = True, save_weights_only = False)
    
    # model fit
    model.fit(training_set ,verbose = 1 ,steps_per_epoch = training_set.n//training_set.batch_size ,validation_data = validation_set ,validation_steps= validation_set.n//validation_set.batch_size ,epochs = EPOCH_SIZE ,callbacks = [best_model,early])
    
    # fine tuning
    pretrained_model.trainable = True
    
    # print model summary
    pretrained_model.summary()
    
    # optimizers choices
    opt1 = SGD(learning_rate = 0.01, decay=0.00001)
    
    model.compile(loss = 'categorical_crossentropy', optimizer = opt1, metrics = [METRICS,"accuracy"])
    
    model.summary()
    
    # model fit
    history = model.fit(training_set ,verbose = 1 ,steps_per_epoch = training_set.n//training_set.batch_size ,validation_data = validation_set ,validation_steps= validation_set.n//validation_set.batch_size ,epochs = EPOCH_SIZE ,callbacks = [best_model,early])
    
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
    
 
if TRAINING:
    model_evaluation()
    
model = keras.models.load_model(save_best_model)

filenames = test_set.filenames
nb_samples = len(filenames)
preds = model.predict_generator(test_set,steps = nb_samples, verbose = 1)
y_preds = np.argmax(preds, axis = -1)

y_test = test_set.classes

# set plot figure size
fig, c_ax = plt.subplots(1,1, figsize = (12, 8))

class_list = os.listdir(ROOT+"Datasets/classification_data/train_data")
target = []
for i in range(len(class_list)):
    target.append(class_list[i])
    if i == 4:
        break

# function for scoring roc auc score for multi-class
def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    for (idx, c_label) in enumerate(target):
        fpr, tpr, thresholds = roc_curve(y_test[:,idx].astype(int), y_pred[:,idx])
        c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % (c_label, auc(fpr, tpr)))
    c_ax.plot(fpr, fpr, 'b-', label = 'Random Guessing')
    return roc_auc_score(y_test, y_pred, average=average,multi_class='ovo')


print('ROC AUC score:', multiclass_roc_auc_score(y_test, y_preds))

c_ax.legend()
c_ax.set_xlabel('False Positive Rate')
c_ax.set_ylabel('True Positive Rate')
plt.show()
