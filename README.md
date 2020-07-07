# Single Human Pose Estimation Using SVM
## Introduction
The problem of human pose estimation, defined as the problem of localization of human joints. 
Human pose estimation in this work provide the method that not using Deep Learning approach as much as possible.

## Approach
<p align="center">
  <img src="https://raw.githubusercontent.com/MajorTom3K1M/CV_PoseEstimation/master/example/pic1.png">
</p>

#### Foreground Segmentation
- Background Subtraction, Morphology, Canny Edge Detection, Convex Hull
#### Depth Estimation (Deep Learning)
#### Superpixel Generation
- Reduce time used for the classification
#### Superpixel Classification (using SVM)
#### Joint Estimation

## Classification Feature
#### Superpixel Relative Position
- X
- Y
#### Superpixel Depth
- Alpha

## Result
From the implementation has showed that trained SVM classifier will have a good accuracy 
only when the image have a clear depth estimation value.<br/>
In this work estimate 16 key points: Head, Neck, Spine, Pelvis, r_Ankle, l_Ankle, r_Knee, l_Knee, r_Hip, l_Hip, r_Shoulder,
l_Shoulder, r_Elbow, l_Elbow, r_Wrist, l_Wrist

|          |  Head  | l_Shoulder | r_Shoulder | l_Elbow  | r_Elbow | l_Wrist | r_Wrist | Neck   |
| -------- | ------ | ---------- | ---------- | -------- | ------- | ------- | ------- | ------ |
| All      | 10     | 10         | 10         | 10       | 10      | 8       | 9       | 9      |
| Correct  | 7      | 4          | 2          | 2        | 0       | 1       | 1       | 4      |
| %        | 70.00% | 40.00%     | 20.00%     | 20.00%   | 0.00%   | 12.50%  | 11.11%  | 44.44% |

|          |  Spine  | Pelvis | l_Hip  | r_Hip  | l_Knee  | r_Knee  | l_Ankle | r_Ankle |
| -------- | ------- | ------ | ------ | ------ | ------- | ------- | ------- | ------- |
| All      | 9       | 8      | 10     | 10     | 10      | 10      | 9       | 8       |
| Correct  | 8       | 3      | 5      | 4      | 0       | 0       | 2       | 2       |
| %        | 88.88%  | 38.00% | 50.00% | 40.00% | 0.00%   | 0.00%   | 22.22%  | 25.00%  |

## Result Example
<p align="center">
  <img src="https://raw.githubusercontent.com/MajorTom3K1M/CV_PoseEstimation/master/example/pic2.png">
</p>

