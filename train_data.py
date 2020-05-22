import csv
import sys
import os
import cv2
import random
import glob
import enum
import joblib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from sklearn import svm
from skimage import segmentation, color
from skimage.measure import regionprops

class Body(enum.Enum):
    r_Ankle = 1
    r_Knee = 2
    r_Hip = 3
    l_Hip = 4
    l_Knee = 5
    l_Ankle = 6
    b_Pelvis = 7
    b_Spine = 8
    b_Neck = 9
    b_Head = 10
    r_Wrist = 11
    r_Elbow = 12
    r_Shoulder = 13
    l_Shoulder = 14
    l_Elbow = 15
    l_Wrist = 16
    null = 0

def dataLoader():
    # csv_path = 'train_datasets2.csv' 
    csv_path = 'csv_file/train_data2.csv' 

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        recs = []
        for row in reader:
            recs.append(row)
        return recs

readdepthimg = "testsets/100238_533980.jpg"
readogimg = "image/100238_533980.jpg"
img = cv2.imread(readdepthimg)
img_segments = segmentation.slic(img, n_segments=50, sigma=5)
superpixels = color.label2rgb(img_segments, img, kind='avg')
gray_image = cv2.cvtColor(superpixels, cv2.COLOR_BGR2GRAY)

# plt.imshow(gray_image)
# plt.show()

testsets = []
regions = regionprops(img_segments)
for i ,props in enumerate(regions):
    cy, cx = props.centroid  # centroid coordinates
    v = props.label  # value of label
    height, width, channels = img.shape
    depth_normalize = (gray_image[int(cy)][int(cx)] - min(gray_image.flatten()))/(max(gray_image.flatten()) - min(gray_image.flatten()))
    testsets.append([cx/width,cy/height,depth_normalize])

# plt.imshow(gray_image)
# plt.show()

datasets = dataLoader()

X_train = np.delete(datasets, 3, 1)
Y_train = np.delete(datasets, [0,1,2], 1).reshape(-1)


classifier = svm.SVC(gamma='auto', probability=True, class_weight='balanced')
classifier.fit(X_train, Y_train)

class_probabilities = classifier.predict_proba(testsets)
confidence_score = np.max(class_probabilities, axis=1)
predicted = classifier.predict(testsets)
unique = np.unique(predicted)
body_part_max = np.zeros(17)

for class_data in unique:
    index = np.where(predicted == class_data)
    for i in index:
        body_part_max[int(class_data)] = max(confidence_score[i])

# print("Confidence Scorre", body_part_max[2])
print("Predict", predicted)
print("Class Probabilities", np.max(class_probabilities, axis=1))
print("Find index", np.where(predicted == str(10)))

# save the model to disk
filename = 'finalized_model2.sav'
joblib.dump(classifier, filename)

# font 
font = cv2.FONT_HERSHEY_SIMPLEX 

# fontScale 
fontScale = 1

img_og = cv2.imread(readogimg)
regions = regionprops(img_segments)
for i ,props in enumerate(regions):
    cy, cx = props.centroid  # centroid coordinates
    v = props.label  # value of label
    height, width, channels = img.shape
    if int(predicted[v -1]) > 0:
        if int(predicted[v -1]) == Body.b_Head.value and confidence_score[v - 1] == body_part_max[Body.b_Head.value]:
            cv2.putText(img_og, 'b_Head', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=4, color=(0, 0, 0)) #Black
        elif int(predicted[v -1]) == Body.l_Shoulder.value and confidence_score[v - 1] == body_part_max[Body.l_Shoulder.value]: 
            cv2.putText(img_og, 'l_Shoulder', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 255, 0)) #Green
        elif int(predicted[v -1]) == Body.r_Shoulder.value and confidence_score[v - 1] == body_part_max[Body.r_Shoulder.value]: 
            cv2.putText(img_og, 'r_Shoulder', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 255, 0)) #Green
        elif int(predicted[v -1]) == Body.b_Spine.value and confidence_score[v - 1] == body_part_max[Body.b_Spine.value]: 
            cv2.putText(img_og, 'b_Spine', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(255,0,0)) #RED
        elif int(predicted[v -1]) == Body.l_Ankle.value and confidence_score[v - 1] == body_part_max[Body.l_Ankle.value]: 
            cv2.putText(img_og, 'l_Ankle', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0,220,255)) # Light Blue
        elif int(predicted[v -1]) == Body.r_Ankle.value and confidence_score[v - 1] == body_part_max[Body.r_Ankle.value]: 
            cv2.putText(img_og, 'r_Ankle', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(122,103,83)) #Brown
        elif int(predicted[v -1]) == Body.r_Knee.value and confidence_score[v - 1] == body_part_max[Body.r_Knee.value]: 
            cv2.putText(img_og, 'r_Knee', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(122,103,83)) #Brown
        elif int(predicted[v -1]) == Body.l_Knee.value and confidence_score[v - 1] == body_part_max[Body.l_Knee.value]: 
            cv2.putText(img_og, 'l_Knee', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Hip.value and confidence_score[v - 1] == body_part_max[Body.r_Hip.value]: 
            cv2.putText(img_og, 'r_Hip', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Hip.value and confidence_score[v - 1] == body_part_max[Body.l_Hip.value]: 
            cv2.putText(img_og, 'l_Hip', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.b_Pelvis.value and confidence_score[v - 1] == body_part_max[Body.b_Pelvis.value]: 
            cv2.putText(img_og, 'b_Pelvis', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.b_Neck.value and confidence_score[v - 1] == body_part_max[Body.b_Neck.value]: 
            cv2.putText(img_og, 'b_Neck', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Wrist.value and confidence_score[v - 1] == body_part_max[Body.r_Wrist.value]: 
            cv2.putText(img_og, 'r_Wrist', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Wrist.value and confidence_score[v - 1] == body_part_max[Body.l_Wrist.value]: 
            cv2.putText(img_og, 'l_Wrist', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Elbow.value and confidence_score[v - 1] == body_part_max[Body.r_Elbow.value]: 
            cv2.putText(img_og, 'r_Elbow', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Elbow.value and confidence_score[v - 1] == body_part_max[Body.l_Elbow.value]: 
            cv2.putText(img_og, 'l_Elbow', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(img_og, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue

plt.imshow(img_og)
plt.show()