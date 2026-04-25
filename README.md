# Photogrammetric ComputerVision
A deep dive into classical & modern 3D Computer Vision concepts and techniques applied here to the calibrated\synchronized stereo camera imagery provided by the "KITTI Dataset"

## I. DLT Triangulation of Point Pairs from KITTI Stereo Cameras

DLT Triangulation (described by MVG) of corresponding image point-feature pairs.

Please watch: 
[Full Video Recordings on YouTube](https://www.youtube.com/watch?v=w3AwZM1RpVQ&list=PL9IYlUueNFoa8mLsTHtWhH6aflSdcyqWZ&index=1)

![Triangulation Results](./images/DLT_Triangulation_0.png)
![Triangulation Results](./images/DLT_Triangulation_1.png)
Note how the triangulation resolution diminishes rapidly along the Line-of-Sight (Z axis).
Point color represents Z-Axis depth. 

## II. Optical Flow from KITTI Camera Sequance

Optical Flow (described by Ma et al) of a fixed grid of points over a camera image sequence. 
It is Multi-Scale (i.e. recusrsively downsamples a pyramid of images) and applies Gradient based (Lucas and Kanade) computations. 

[Full Video Recordings on YouTube](https://www.youtube.com/watch?v=mm-BLc3SGRY&list=PL9IYlUueNFoa8mLsTHtWhH6aflSdcyqWZ&index=7)

![Optical Flow Results](./images/OpticalFlowOutput_Reduced.png)

## Installation:
Only `numpy, matplotlib, scipy, scikit-image, pytest` packages are required for this script.

To run within a virtual environment, create a separate virtual environment for the new project 

`python3 -m venv .venv` Specifying `.venv` as the directory for it.

`source .venv/bin/activate` Activate the virtual environment by sourcing the activate script.

`pip install -r requirements.txt` Install required packages

## Usage:
`pytest` Run Unit Tests

`python3 main_stereo_dlt_triangulation.py` Run DLT Triangulation over example image pair

`python3 main_optical_flow.py` Run Optical Flow over example image pair

`deactive` Deactivate Virtual Environment before closing terminal.

## ORB-SLAM2 from Universidad de Zaragoza (Raúl Mur-Artal et al)
- **Original Project Page**: https://webdiis.unizar.es/~raulmur/orbslam/
- **Original Code**: https://github.com/raulmur/ORB_SLAM2

Please watch: 
[My ORB-SLAM2 Stereo-KITTI Ouptuts on YouTube](https://www.youtube.com/watch?v=-Z-bCY-UboU&list=PL9IYlUueNFoa8mLsTHtWhH6aflSdcyqWZ&index=10)

Several minor tweaks were made to run on: 
- Ubuntu 24.04.4, C++ 14, opencv 4.6.0, eigen3 3.4.0, Pangolin 0.9.5

Apply `ORBSLAM2RefactorForUpgradedDeps.patch` to your cloned ORB-SLAM2 repo to run locally.

ORB-SLAM computes in real-time the camera trajectory and sparse 3D scene reconstruction for Monocular, Stereo, and RGB-D Cameras. 
It is Keypoint (ORB feature) based, and employs Bundle-Adjustment to close large loops.

Unlike Monocular-SLAM, the Stereo-SLAM estimates the map and trajectory with metric scale and does not suffer from scale drift. 
See below how Stereo ORBSLAM approaches the Loop Closure point with perfect accuracy, while the Monocular ORBSLAM has accumulated significant drift.
![ORB-SLAM2 Monocular vs Stereo](./images/ORB_SLAM2_MonoStereoCompare.png)
Both Mono and Stereo detect the Loop Closure and refine the total Map via Bundle Adjustment upon detection.


## Resources:
Geiger A, Lenz P, Stiller C, Urtasun R, _Vision meets Robotics: The KITTI Dataset_, International Journal of Robotics Research (IJRR), 2013, https://www.cvlibs.net/datasets/kitti/raw_data.php

Hartley R, Zisserman A,_Multiple View Geometry in Computer Vision_, 2003, Cambridge University Press, 2nd edition

Ma Y, Soatto S, Kosecká, J, & Sastry S S (2004). _An Invitation to 3-D Vision: From Images to Geometric Models_. Springer-Verlag.

Qian-Yi Zhou and Jaesik Park and Vladlen Koltun, _{Open3D}: {A} Modern Library for {3D} Data Processing_, arXiv:1801.09847, 2018

Raúl Mur-Artal, and Juan D. Tardós.
_ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras_
ArXiv preprint arXiv:1610.06475, 2016.

Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós.
_ORB-SLAM: A Versatile and Accurate Monocular SLAM System._
IEEE Transactions on Robotics, vol. 31, no. 5, pp. 1147-1163, October 2015.
(2015 IEEE Transactions on Robotics Best Paper Award)

Torralba, A. and Isola, P. and Freeman, W.T. _Foundations of Computer Vision_, 2024, Adaptive Computation and Machine Learning series, MIT Press, https://mitpress.mit.edu/9780262048972/foundations-of-computer-vision/
