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
from sklearn import model_selection
from sklearn import svm
from skimage import segmentation, color
from skimage.measure import regionprops

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

# read_img = "testsets/11411_445875.jpg"
# original_img = "test_image/11411_445875.jpg"
# read_img = "testsets/109095_199654.jpg"
# original_img = "test_image/109095_199654.jpg"
# read_img = "testsets/731_428532.jpg"
# original_img = "test_image/731_428532.jpg"
# read_img = "testsets/731_428532.jpg"
# original_img = "test_image/731_428532.jpg"
# read_img = "testsets/105545_2155923.jpg"
# original_img = "test_image/105545_2155923.jpg"
# read_img = "testsets/10591_1247286.jpg"
# original_img = "test_image/10591_1247286.jpg"
# read_img = "testsets/107782_463132.jpg"
# original_img = "test_image/107782_463132.jpg"
# read_img = "testsets/106810_198201.jpg"
# original_img = "test_image/106810_198201.jpg"
# read_img = "testsets/103250_524128.jpg"
# original_img = "test_image/103250_524128.jpg"
# read_img = "image/depth/100238_533980.jpg"
# original_img = "image/100238_533980.jpg"

img = cv2.imread(read_img)
img_segments = segmentation.slic(img, n_segments=50, sigma=5)
superpixels = color.label2rgb(img_segments, img, kind='avg')
gray_image = cv2.cvtColor(superpixels, cv2.COLOR_BGR2GRAY)

testsets = []
regions = regionprops(img_segments)
for i ,props in enumerate(regions):
    cy, cx = props.centroid  # centroid coordinates
    v = props.label  # value of label
    height, width, channels = img.shape
    depth_normalize = (gray_image[int(cy)][int(cx)] - min(gray_image.flatten()))/(max(gray_image.flatten()) - min(gray_image.flatten()))
    testsets.append([cx/width,cy/height,depth_normalize])

filename = "checkpoints/finalized_model.sav"
classifier = joblib.load(filename)
predicted = classifier.predict(testsets)
class_probabilities = classifier.predict_proba(testsets)
confidence_score = np.max(class_probabilities, axis=1)
predicted = classifier.predict(testsets)
unique = np.unique(predicted)
body_part_max = np.zeros(17)

for class_data in unique:
    index = np.where(predicted == class_data)
    for i in index:
        body_part_max[int(class_data)] = max(confidence_score[i])

print("Predicted : " , predicted)

joints = [None] * 3 * 16
# joints = np.array([None] * 3 * 16, dtype=float)

font = cv2.FONT_HERSHEY_SIMPLEX 
fontScale = 1


read_original_img = cv2.imread(original_img)
regions = regionprops(img_segments)
for i ,props in enumerate(regions):
    cy, cx = props.centroid  # centroid coordinates
    v = props.label  # value of label
    height, width, channels = img.shape
    if int(predicted[v -1]) > 0:
        if int(predicted[v -1]) == Body.b_Head.value and confidence_score[v - 1] == body_part_max[Body.b_Head.value]:

            joints[(Body.b_Head.value - 1)*3] = cx
            joints[(Body.b_Head.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'b_Head', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=4, color=(0, 0, 0)) #Black
        elif int(predicted[v -1]) == Body.l_Shoulder.value and confidence_score[v - 1] == body_part_max[Body.l_Shoulder.value]: 

            joints[(Body.l_Shoulder.value - 1)*3] = cx
            joints[(Body.l_Shoulder.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Shoulder', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 255, 0)) #Green
        elif int(predicted[v -1]) == Body.r_Shoulder.value and confidence_score[v - 1] == body_part_max[Body.r_Shoulder.value]: 

            joints[(Body.r_Shoulder.value - 1)*3] = cx
            joints[(Body.r_Shoulder.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Shoulder', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 255, 0)) #Green
        elif int(predicted[v -1]) == Body.b_Spine.value and confidence_score[v - 1] == body_part_max[Body.b_Spine.value]: 

            joints[(Body.b_Spine.value - 1)*3] = cx
            joints[(Body.b_Spine.value - 1)*3 + 1] = cy
            
            cv2.putText(read_original_img, 'b_Spine', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(255,0,0)) #RED
        elif int(predicted[v -1]) == Body.l_Ankle.value and confidence_score[v - 1] == body_part_max[Body.l_Ankle.value]: 

            joints[(Body.l_Ankle.value - 1)*3] = cx
            joints[(Body.l_Ankle.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Ankle', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0,220,255)) # Light Blue
        elif int(predicted[v -1]) == Body.r_Ankle.value and confidence_score[v - 1] == body_part_max[Body.r_Ankle.value]: 

            joints[(Body.r_Ankle.value - 1)*3] = cx
            joints[(Body.r_Ankle.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Ankle', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(122,103,83)) #Brown
        elif int(predicted[v -1]) == Body.r_Knee.value and confidence_score[v - 1] == body_part_max[Body.r_Knee.value]: 

            joints[(Body.r_Knee.value - 1)*3] = cx
            joints[(Body.r_Knee.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Knee', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(122,103,83)) #Brown
        elif int(predicted[v -1]) == Body.l_Knee.value and confidence_score[v - 1] == body_part_max[Body.l_Knee.value]: 

            joints[(Body.l_Knee.value - 1)*3] = cx
            joints[(Body.l_Knee.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Knee', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Hip.value and confidence_score[v - 1] == body_part_max[Body.r_Hip.value]: 

            joints[(Body.r_Hip.value - 1)*3] = cx
            joints[(Body.r_Hip.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Hip', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Hip.value and confidence_score[v - 1] == body_part_max[Body.l_Hip.value]: 

            joints[(Body.l_Hip.value - 1)*3] = cx
            joints[(Body.l_Hip.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Hip', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.b_Pelvis.value and confidence_score[v - 1] == body_part_max[Body.b_Pelvis.value]: 

            joints[(Body.b_Pelvis.value - 1)*3] = cx
            joints[(Body.b_Pelvis.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'b_Pelvis', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.b_Neck.value and confidence_score[v - 1] == body_part_max[Body.b_Neck.value]: 

            joints[(Body.b_Neck.value - 1)*3] = cx
            joints[(Body.b_Neck.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'b_Neck', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Wrist.value and confidence_score[v - 1] == body_part_max[Body.r_Wrist.value]: 

            joints[(Body.r_Wrist.value - 1)*3] = cx
            joints[(Body.r_Wrist.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Wrist', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Wrist.value and confidence_score[v - 1] == body_part_max[Body.l_Wrist.value]: 

            joints[(Body.l_Wrist.value - 1)*3] = cx
            joints[(Body.l_Wrist.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Wrist', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.r_Elbow.value and confidence_score[v - 1] == body_part_max[Body.r_Elbow.value]: 

            joints[(Body.r_Elbow.value - 1)*3] = cx
            joints[(Body.r_Elbow.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'r_Elbow', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue
        elif int(predicted[v -1]) == Body.l_Elbow.value and confidence_score[v - 1] == body_part_max[Body.l_Elbow.value]: 

            joints[(Body.l_Elbow.value - 1)*3] = cx
            joints[(Body.l_Elbow.value - 1)*3 + 1] = cy

            cv2.putText(read_original_img, 'l_Elbow', (int(cx),int(cy)), font, 0.4, thickness=1, color=(255,0,0))
            draw = cv2.circle(read_original_img, (int(cx),int(cy)), radius=3, color=(0, 0, 255)) #Blue

for i, data in enumerate(joints):
    if None == data:
        if i - 1 > 0 and i + 1 < len(joints) - 1:
            if joints[i - 1] is not None and joints[i + 1] is not None:
                joints[i] = 0
    elif i == (Body.b_Head.value - 1) * 3 and joints[(Body.b_Head.value - 1) * 3] is not None:
        joints[i + 2] = 0

def plot_joint(rec, img_folder):
    print('Image at: ' + img_folder)
	
    img = cv2.imread(img_folder)

    r = 5
    bombs = [[0,1],[1,2]
            ,[3,4],[4,5]
            ,[6,7],[6,3]
            ,[6,2],[7,8]
            ,[8,9],[8,12],[8,13]
            ,[10,11],[11,12]
            ,[13,14],[14,15] ]
    colors = [(255,0,0),(255,0,0),
                (0,255,0),(0,255,0),
                (0,0,255),(181,239,16),
                (181,239,16),(0,0,255),
                (0,0,255),(255,216,202),(255,216,202),
                (128,128,0),(128,128,0),
                (128,0,128),(128,0,128)]
    r = 5 
    for b_id in range(len(bombs)):
        b = bombs[b_id]
        color = colors[b_id]

        x1 = rec[ b[0] * 3 ]
        y1 = rec[ b[0] * 3 + 1]
        v1 = rec[ b[0] * 3 + 2]

        x2 = rec[ b[1] * 3 ]
        y2 = rec[ b[1] * 3 + 1]
        v2 = rec[ b[1] * 3 + 2]

        if v1 != None and v2 != None:
            cv2.line(img, (int(x1),int(y1)), (int(x2),int(y2)), color=color, thickness=5)
        elif v1 != None:
            cv2.circle(img, (int(x1), int(y1)), radius=r, color=color, thickness=cv2.FILLED)
        elif v2 != None:
            cv2.circle(img, (int(x2), int(y2)), radius=r, color=color, thickness=cv2.FILLED)

    _ , axs = plt.subplots(1,2)
    axs[1].imshow(read_original_img)
    axs[0].imshow(img)
    plt.show()

plot_joint(joints, original_img)


# plt.imshow(read_original_img)
# plt.show()