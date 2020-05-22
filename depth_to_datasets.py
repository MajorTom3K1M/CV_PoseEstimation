import cv2
import os
import csv
import sys
import random
import glob
import matplotlib.pyplot as plt 
import numpy as np

from skimage import segmentation, color
from skimage.measure import regionprops

# img_dir = "outfile/"
# out_path = "datasets/"
# data_path = os.path.join(img_dir, '*g')
# files = glob.glob(data_path)
# data = []

def assignLabel(keypoint, segment_no, img_segments, img_no):
    r_Ankle = [keypoint[img_no][0*3], keypoint[img_no][0*3+1]] if keypoint[img_no][0*3+2] != 'nan' else 'nan'
    r_Knee = [keypoint[img_no][1*3], keypoint[img_no][1*3+1]] if keypoint[img_no][1*3+2] != 'nan' else 'nan'
    r_Hip = [keypoint[img_no][2*3], keypoint[img_no][2*3+1]] if keypoint[img_no][2*3+2] != 'nan' else 'nan'
    l_Hip = [keypoint[img_no][3*3], keypoint[img_no][3*3+1]] if keypoint[img_no][3*3+2] != 'nan' else 'nan'
    l_Knee = [keypoint[img_no][4*3], keypoint[img_no][4*3+1]] if keypoint[img_no][4*3+2] != 'nan' else 'nan'
    l_Ankle = [keypoint[img_no][5*3], keypoint[img_no][5*3+1]] if keypoint[img_no][5*3+2] != 'nan' else 'nan'
    b_Pelvis = [keypoint[img_no][6*3], keypoint[img_no][6*3+1]] if keypoint[img_no][6*3+2] != 'nan' else 'nan'
    b_Spine = [keypoint[img_no][7*3], keypoint[img_no][7*3+1]] if keypoint[img_no][7*3+2] != 'nan' else 'nan'
    b_Neck = [keypoint[img_no][8*3], keypoint[img_no][8*3+1]] if keypoint[img_no][8*3+2] != 'nan' else 'nan'
    b_Head = [keypoint[img_no][9*3], keypoint[img_no][9*3+1]] if keypoint[img_no][9*3+2] != 'nan' else 'nan'
    r_Wrist = [keypoint[img_no][10*3], keypoint[img_no][10*3+1]] if keypoint[img_no][10*3+2] != 'nan' else 'nan'
    r_Elbow = [keypoint[img_no][11*3], keypoint[img_no][11*3+1]] if keypoint[img_no][11*3+2] != 'nan' else 'nan'
    r_Shoulder = [keypoint[img_no][12*3], keypoint[img_no][12*3+1]] if keypoint[img_no][12*3+2] != 'nan' else 'nan'
    l_Shoulder = [keypoint[img_no][13*3], keypoint[img_no][13*3+1]] if keypoint[img_no][13*3+2] != 'nan' else 'nan'
    l_Elbow = [keypoint[img_no][14*3], keypoint[img_no][14*3+1]] if keypoint[img_no][14*3+2] != 'nan' else 'nan'
    l_Wrist = [keypoint[img_no][15*3], keypoint[img_no][15*3+1]] if keypoint[img_no][15*3+2] != 'nan' else 'nan'

    height, width = img_segments.shape

    if r_Ankle != 'nan' and not (int(r_Ankle[1]) > height) and not (int(r_Ankle[0]) > width) and img_segments[int(r_Ankle[1]) - 1][int(r_Ankle[0]) - 1] == segment_no:
        return 1
    elif r_Knee != 'nan' and not (int(r_Knee[1])  > height) and not (int(r_Knee[0]) > width)  and img_segments[int(r_Knee[1]) - 1][int(r_Knee[0]) - 1] == segment_no:
        return 2
    elif r_Hip != 'nan' and not (int(r_Hip[1]) > height) and not (int(r_Hip[0]) > width) and img_segments[int(r_Hip[1]) - 1][int(r_Hip[0]) - 1] == segment_no:
        return 3
    elif l_Hip != 'nan' and not (int(l_Hip[1]) > height) and not (int(l_Hip[0]) > width) and img_segments[int(l_Hip[1]) - 1][int(l_Hip[0]) - 1] == segment_no:
        return 4
    elif l_Knee != 'nan' and not (int(l_Knee[1]) > height) and not (int(l_Knee[0]) > width) and img_segments[int(l_Knee[1]) - 1][int(l_Knee[0]) - 1] == segment_no:
        return 5
    elif l_Ankle != 'nan' and not (int(l_Ankle[1])> height) and not (int(l_Ankle[0]) > width) and img_segments[int(l_Ankle[1])- 1][int(l_Ankle[0])- 1] == segment_no:
        return 6
    elif b_Pelvis != 'nan' and not (int(b_Pelvis[1]) > height) and not (int(b_Pelvis[0]) > width) and img_segments[int(b_Pelvis[1])- 1][int(b_Pelvis[0])- 1] == segment_no:
        return 7
    elif b_Spine != 'nan' and not (int(b_Spine[1]) > height) and not (int(b_Spine[0]) > width) and img_segments[int(b_Spine[1])- 1][int(b_Spine[0])- 1] == segment_no:
        return 8
    elif b_Neck != 'nan' and not (int(b_Neck[1]) > height) and not (int(b_Neck[0]) > width) and img_segments[int(b_Neck[1])- 1][int(b_Neck[0])- 1] == segment_no:
        return 9
    elif b_Head != 'nan' and not (int(b_Head[1]) > height) and not (int(b_Head[0]) > width) and img_segments[int(b_Head[1])- 1][int(b_Head[0])- 1] == segment_no:
        return 10
    elif r_Wrist != 'nan' and not (int(r_Wrist[1]) > height) and not (int(r_Wrist[0]) > width) and img_segments[int(r_Wrist[1])- 1][int(r_Wrist[0])- 1] == segment_no:
        return 11
    elif r_Elbow != 'nan' and not (int(r_Elbow[1]) > height) and not (int(r_Elbow[0]) > width) and img_segments[int(r_Elbow[1])- 1][int(r_Elbow[0])- 1] == segment_no:
        return 12
    elif r_Shoulder != 'nan' and not (int(r_Shoulder[1]) > height) and not (int(r_Shoulder[0]) > width) and img_segments[int(r_Shoulder[1])- 1][int(r_Shoulder[0])- 1] == segment_no:
        return 13
    elif l_Shoulder != 'nan' and not (int(l_Shoulder[1]) > height) and not (int(l_Shoulder[0]) > width)  and img_segments[int(l_Shoulder[1])- 1][int(l_Shoulder[0])- 1] == segment_no:
        return 14
    elif l_Elbow != 'nan' and not (int(l_Elbow[1]) > height) and not (int(l_Elbow[0]) > width)  and img_segments[int(l_Elbow[1])- 1][int(l_Elbow[0])- 1] == segment_no:
        return 15
    elif l_Wrist != 'nan' and not (int(l_Wrist[1]) > height) and not (int(l_Wrist[0]) > width)  and img_segments[int(l_Wrist[1])- 1][int(l_Wrist[0])- 1] == segment_no:
        return 16
    else:
        return 0

def writeCSV(filename ,data): 
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(data)

# img = cv2.imread("outfile/117908_208247.jpg")
# # img = cv2.imread("outfile/1000_1234574.jpg")

# img_segments = segmentation.slic(img, n_segments=50, sigma=5)
# superpixels = color.label2rgb(img_segments, img, kind='avg')
# gray_image = cv2.cvtColor(superpixels, cv2.COLOR_BGR2GRAY)

# cv2.imwrite(out_path + "test.jpg", gray_image)

# for f1 in files:
#     img = cv2.imread(f1)
#     img_segments = segmentation.slic(img, compactness=20, n_segments=200)
#     superpixels = color.label2rgb(img_segments, img, kind='avg')
#     gray_image = cv2.cvtColor(superpixels, cv2.COLOR_BGR2GRAY)
#     cv2.imwrite(out_path + f1.split('\\')[-1], gray_image)
#     print("Write", out_path + f1.split('\\')[-1])
    # data.append(gray_image)

# print(data)

# plt.imshow(gray_image)
# plt.show()

# Load Original Data
def dataLoader():
    # csv_path = 'csv_file/lip_train_set.csv' 
    csv_path = 'csv_file/lip_val_set.csv' 
    img_dir = "testsets/"
    data_path = os.path.join(img_dir, '*g')
    files = glob.glob(data_path)

    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        files_name = []
        recs = []
        for i in range(len(files)):
            files_name.append(files[i].split("\\")[-1])

        for row in reader:
            if row[0] in files_name:
                recs.append(row)
        # random_id = random.randint(0, len(files) - 1)
        # plot_joint(recs[random_id], img_root['train'])  
        return recs, files

recs, files = dataLoader()
keypoint = np.delete(recs, 0, 1)

datasets = []

# print(files)
# for i ,k in enumerate(keypoint):
#     print(k[0], files[i])

for img_id ,f in enumerate(files):
    img = cv2.imread(f)
    # img = cv2.imread("outfile/1000_1234574.jpg")

    img_segments = segmentation.slic(img, n_segments=50, sigma=5)
    superpixels = color.label2rgb(img_segments, img, kind='avg')
    gray_image = cv2.cvtColor(superpixels, cv2.COLOR_BGR2GRAY)

    # Make Datasets
    print(img_id, f)
    if img_id == 1:
        plt.imshow(gray_image)
        plt.show()

    regions = regionprops(img_segments)
    for i ,props in enumerate(regions):
        cy, cx = props.centroid  # centroid coordinates
        v = props.label  # value of label
        height, width, channels = img.shape
        depth_normalize = (gray_image[int(cy)][int(cx)] - min(gray_image.flatten()))/(max(gray_image.flatten()) - min(gray_image.flatten()))
        label = assignLabel(keypoint,v,img_segments, img_id)
        datasets.append([cx/width,cy/height,depth_normalize,label])

    
writeCSV('csv_file/test_datasets222.csv', datasets)
print('FINISH!')