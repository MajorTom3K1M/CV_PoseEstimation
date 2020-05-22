import sys
import csv
import os
import cv2
import random
import glob
import enum
import joblib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from sklearn import model_selection
from sklearn import svm
from skimage import segmentation, color
from skimage.measure import regionprops
from sklearn.metrics import balanced_accuracy_score, accuracy_score

# Class Label
# 1,R_Ankle
# 2,R_Knee
# 3,R_Hip
# 4,L_Hip
# 5,L_Knee
# 6,L_Ankle
# 7,B_Pelvis
# 8,B_Spine
# 9,B_Neck
# 10,B_Head
# 11,R_Wrist
# 12,R_Elbow
# 13,R_Shoulder
# 14,L_Shoulder
# 15,L_Elbow
# 16,L_Wrist
# 0,Null
def dataLoader():
    csv_path = 'csv_file/test_datasets.csv' 

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        recs = []
        for row in reader:
            recs.append(row)
        return recs

datasets = dataLoader()

X_test = np.delete(datasets, 3, 1)
Y_true = np.delete(datasets, [0,1,2], 1).reshape(-1)

filename = "checkpoints/finalized_model.sav"
classifier = joblib.load(filename)
Y_pred = classifier.predict(X_test)
class_probabilities = classifier.predict_proba(X_test)

score = accuracy_score(Y_true, Y_pred)
print("Accuracy score: {0:.2f} %".format(100 * score))

score = balanced_accuracy_score(Y_true, Y_pred)
print("Balanced Accuracy score: {0:.2f} %".format(100 * score))
