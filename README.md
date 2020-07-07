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
