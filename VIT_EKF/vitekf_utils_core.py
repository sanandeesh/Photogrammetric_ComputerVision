"""
    Core Utilities
    Supplements VIT-EKF by providing utilities for image extraction, processing, triangulation, etc.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import skimage as ski # Template Matching
from types import SimpleNamespace

# Calibrated Camera Matrixes (Stereo Pair, 0.54m apart), from 2011_09_26 recording
P_Left = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                   [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

P_Right = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02],
                    [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

# params for Shi-Tomasi corner detection
feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def GetImagePair(data_path, index):
    """Get KITTI Left\Right GreyScale Image Pair
       
    Args:
        data_path: path to root directory of the recording.
        index: index of sample withing recording

    Returns:
        Left & Right Image
    """
    width = 10
    padded_number = str(index).zfill(width)
    print(padded_number)
    #data_path = '../' + data_path
    image_name_left   = data_path + 'image_00/data/' + padded_number + '.png'
    image_name_right  = data_path + 'image_01/data/' + padded_number + '.png'
    # 
    image_left = plt.imread(image_name_left)
    image_left = (image_left * 255).astype(np.uint8)
    image_right = plt.imread(image_name_right)
    image_right = (image_right * 255).astype(np.uint8)
    #
    images = SimpleNamespace()
    images.left = image_left
    images.right = image_right

    return images

kernel_half_length = 4 # Template-Matching Window Half-Size (Pixels)
def ComputePointCorrespondences(left_image, right_image, left_poi):
    """Compute corresponding pixels in right image, given features in left image.
       
    Args:
        left_image: 
        right_image:
        left_poi: points-of-interest in left image (2,N)  

    Returns:
        Corresponding points on right image.
    """
    [num_rows_img, num_cols_img] = left_image.shape
    left_poi_rows = left_poi[1,:] # Y - Row
    left_poi_cols = left_poi[0,:] # X - Col
    num_points = left_poi.shape[1] # len(left_poi_rows)
    right_poi_rows = np.zeros((num_points,))
    right_poi_cols = np.zeros((num_points,))
    k = kernel_half_length
    for i in range(num_points):
        right_poi_rows[i] = left_poi_rows[i]
        kernel_rows = max(left_poi_rows[i] - k, 0), min(left_poi_rows[i] + k + 1, num_rows_img-1)
        kernel_cols = max(left_poi_cols[i] - k, 0), min(left_poi_cols[i] + k + 1, num_cols_img-1)
        kernel = left_image[kernel_rows[0]:kernel_rows[1], kernel_cols[0]:kernel_cols[1]]
        start_col = max(left_poi_cols[i] - 100, 0) 
        target_img_roi = right_image[kernel_rows[0]:kernel_rows[1], start_col:kernel_cols[1]]
        # 
        correlate_result = ski.feature.match_template(target_img_roi, kernel, pad_input=True)
        right_poi_cols[i] = np.argmax(correlate_result[k,:]) + start_col
    #
    correspondence_pixels = np.vstack((right_poi_cols,  # X
                                       right_poi_rows)) # Y
    
    return correspondence_pixels

def TriangulateFeaturesRemoveInvalidPoints(pts_2d_left, 
                                           pts_2d_right, 
                                           homog_coord_thresh = 0.001):
    """Triangulate 2D point pairs from Left\Right images into 3D space Remove points too close to infinity.
       
    Args:
        pts_2d_left: 
        pts_2d_right:
        homog_coord_thresh:

    Returns:
        Homogenous 3D Points [4xN] ready for plotting.
    """
    assert ((pts_2d_left.shape == pts_2d_right.shape) and 
            (pts_2d_left.shape[0] == 2))
    #
    points_4d_homo = cv2.triangulatePoints(P_Left, P_Right, pts_2d_left, pts_2d_right)
    # 3. Extract the 4th coordinate (w) for all points
    w_coordinates = points_4d_homo[3, :]
    # 4. Create a boolean mask where |w| >= 0.1
    valid_mask = np.abs(w_coordinates) >= homog_coord_thresh
    # 5. Filter out invalid points from the homogeneous array
    filtered_homo = points_4d_homo[:, valid_mask]
    # 6. Convert remaining valid points to 3D Euclidean space (shape: 3 x M)
    points_3d = filtered_homo / filtered_homo[3, :]
    pts_2d_left  = pts_2d_left[:,valid_mask]
    pts_2d_right = pts_2d_right[:,valid_mask]
    # 
    # Flip Points behind Camera
    i_negative_points = points_3d[2,:] < 0
    #print(points_3d[2:,i_negative_points])
    points_3d[:-1,i_negative_points] = -1*points_3d[:-1,i_negative_points]

    assert ((points_3d.shape[0] == 4) and 
            (points_3d.shape[1] == pts_2d_left.shape[1]) and 
            (pts_2d_left.shape == pts_2d_right.shape))

    return points_3d, pts_2d_left

def GetFeatureROIMask(image):
    """Binary Mask defines search region for Features.
       
    Args:
        image: defines shape of ROI-Mask

    Returns:
        Binary Mask
    """
    feature_roi_mask = np.zeros_like(image)
    feature_roi_mask[7:180,100:] = 1

    return feature_roi_mask

def ProjectPoints(points_3d, P_Cam):
    """Project Points from Homogenous-3D to Homogenous-2D
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates
        P_Cam: [3x4] Projection\Camera Matrix

    Returns:
        Homogenous 2D Points [3xN] ready for plotting.
    """
    points_2d = P_Cam @ points_3d
    points_2d = points_2d / points_2d[-1,:] # Normalize by dividing by Homogenous Coordinate
    #print('Projected points_2d.shape ', points_2d.shape)

    return points_2d

def AssertTracksShape(tracks):
    """Assert Tracks Shape. (Must contain equal number of samples)
       
    Args:
        tracks: 

    Returns:
        Number of Samples
    """    
    num_points = tracks.pts_2d_left.shape[1]
    assert (tracks.pts_2d_left.shape[0] == 2)
    assert (tracks.pts_3d_mean.shape == (4, num_points))
    assert (tracks.pts_3d_cov.shape  == (num_points, 3, 3))
    
    return num_points

# Based on Thrun Motion Model
def GetOdometryErrorParameters():
    """Odometry Error Parameters. Based on Motion Model from `Thrun et al Probabilistic Robotics`
       
    Args:

    Returns:
        Error Params
    """
    err_params = SimpleNamespace()
    # Translational error from translational speed
    err_params.alpha1 = 0.05 # 0.05 to 0.1 (Unitless) Represents a 5% to 10% error in distance traveled.
    # Translational error from rotational speed 
    # Models how much turning the wheels on a spot accidentally causes the robot to shift linearly.
    err_params.alpha2 = 0.001 # 0.001 to 0.01 (m ⋅ s / rad). 
    # Rotational error from translational speed: 
    # Models how much driving forward causes the robot to veer off course due to uneven wheel diameters or floor surfaces.
    err_params.alpha3 = 0.05 # 0.05 to 0.2 (rad / m)
    # Rotational error from rotational speed: 
    # Represents a 5% to 10% error in the execution of a pure turn.
    err_params.alpha4 = 0.05 # 0.05 to 0.1 (Unitless) 

    return err_params

# Based on Thrun Motion Model
def ComputeOdometryCovariance(vf, 
                              vr, 
                              wu, 
                              wl,
                              delta_seconds):
    """Compute Odometry Covariance Matrix as a function of the Odometry values.
    Based on Motion Model from `Thrun et al Probabilistic Robotics`
       
    Args:
        vf: Forward Velocity
        vr: Rightward Velocity 
        wu: Yaw Velocity (Up-Axis)
        wl: Pitch Velocity (Left-Axis)
        delta_seconds: seconds elapsed within this time-step

    Returns:
        Odometry Covariance Matrix
    """
    err = GetOdometryErrorParameters()
    Q_input_space = np.eye(4)
    # Variance in Lateral Displacement (X)
    Q_input_space[0,0] = (err.alpha1*(vr**2) + err.alpha2*(wu**2))*(delta_seconds**2) 
    # Variance in Longitudinal Displacement (Z)
    Q_input_space[1,1] = (err.alpha1*(vf**2) + err.alpha2*(wu**2))*(delta_seconds**2) 
    # Variance in Yaw Displacement (theta)
    Q_input_space[2,2] = (err.alpha3*(vf**2) + err.alpha4*(wu**2))*(delta_seconds**2)  
    # Variance in Pitch Displacement (theta)
    Q_input_space[3,3] = (err.alpha3*(vf**2) + err.alpha4*(wl**2))*(delta_seconds**2)  

    return Q_input_space

"""
See `vitekf_compute_jacobians.py` for details
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ State Transition Model --------------------------------------------------- 
⎡1.0      0.0            0.0     ⎤ ⎡cos(δ_yaw)   0.0  sin(δ_yaw)⎤ ⎡ -δₓ + μₓ ⎤
⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
⎢0.0  cos(δ_pitch)  -sin(δ_pitch)⎥⋅⎢    0.0      1.0     0.0    ⎥⋅⎢   μ_y    ⎥
⎢                                ⎥ ⎢                            ⎥ ⎢          ⎥
⎣0.0  sin(δ_pitch)  cos(δ_pitch) ⎦ ⎣-sin(δ_yaw)  0.0  cos(δ_yaw)⎦ ⎣-δ_z + μ_z⎦

--------------------------------------------------------------------------------------------------------------------
------------ The Jacobian of the State Transition Model wrt "Control Inputs", `δ_x, δ_z, δ_yaw, δ_pitch`------------ 

⎡-1.0  0   -1.0⋅δ_z + 1.0⋅μ_z      0    ⎤
⎢                                       ⎥
⎢ 0    0           0           δ_z - μ_z⎥
⎢                                       ⎥
⎣ 0    -1       δₓ - μₓ         1.0⋅μ_y ⎦
--------------------------------------------------------------------------------------------------------------------
"""

def ComputeProcessNoise(Q_input_space,
                        pts_3d_mean,
                        delta_forward,
                        delta_right):
    """Compute Process Noise by projecting Odometry-Covariance into State-Space via Jacobian.
       
    Args:
        Q_input_space: Odometry-Covariance Matrix 
        pts_3d_mean: 3D Point Coordinate
        delta_forward: 
        delta_right:

    Returns:
        Process Noise Matrix.
    """
    assert pts_3d_mean.shape == (3,)
    x = pts_3d_mean[0]
    y = pts_3d_mean[1] 
    z = pts_3d_mean[2]
    # 
    G = np.array([[-1.0,  0.0, (z -delta_forward),        0.0         ],  
                  [ 0.0,  0.0,         0.0,        (delta_forward - z)],   
                  [ 0.0, -1.0,    (delta_right-x),      y             ]]) 
    Q_state_space = G @ Q_input_space @ G.T

    return Q_state_space

def Assert3DMeanShape(pts_3d_mean):
    assert pts_3d_mean.shape[0] == 4 # (x,y,z, 1.0)
    num_points = pts_3d_mean.shape[1]

    return num_points


def ComputeDistributionsOf3DPoints(pts_3d_mean):
    """ Initialize Covariance Matrices of 3D Points as ellipsoids "extruded along" the Pixel 3D Line-of-Sight
    Depth variance as a function of depth: Variance(z) = z**4/(387.5744**2)*sigma_d**2
    where sigma_d = 0.5 or 1.0 (pixel matching variance)
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates

    Returns:
        pts_3d_cov [N, 3, 3]
    """
    num_points = Assert3DMeanShape(pts_3d_mean)
    pts_3d_x = pts_3d_mean[0,:].reshape(1,-1)
    pts_3d_y = pts_3d_mean[1,:].reshape(1,-1)
    pts_3d_z = pts_3d_mean[2,:].reshape(1,-1)
    # Line-of-Sight Angles
    los_yaw = np.atan(pts_3d_x/pts_3d_z)
    los_el  = np.atan(pts_3d_y/pts_3d_z)
    # 
    pix_match_var = 0.5
    pts_3d_z = pts_3d_mean[2,:].reshape(1,-1)
    z_variance = (pts_3d_z**4)/(387.5744**2)*pix_match_var**2/(np.cos(los_yaw)*np.cos(los_el))
    x_variance = (pts_3d_z/100.0)
    y_variance = x_variance
    pts_3d_var = np.concat((x_variance, 
                            y_variance, 
                            z_variance), axis=0) # [Var(x),Var(y),Var(z)]
    pts_3d_cov_unrot = np.eye(3) * pts_3d_var.T[:, :, np.newaxis]
    #
    R_y = np.zeros((num_points, 3, 3))
    R_x = np.zeros((num_points, 3, 3)) 
    for i_point in np.arange(num_points):
        yaw = los_yaw[0,i_point]
        el = los_el[0,i_point]
        R_y[i_point,:,:] = np.array([[ np.cos(yaw), 0.0, np.sin(yaw)],   # Clockwise
                                     [         0.0, 1.0,         0.0],   
                                     [-np.sin(yaw), 0.0, np.cos(yaw)]]) 
        R_x[i_point,:,:] = np.array([[1.0,         0.0,        0.0],   # Clockwise
                                     [0.0,  np.cos(el), np.sin(el)],   
                                     [0.0, -np.sin(el), np.cos(el)]]) 
        
    pts_3d_cov = R_x @ R_y @ pts_3d_cov_unrot @ np.transpose(R_y, (0, 2, 1)) @ np.transpose(R_x, (0, 2, 1))

    return pts_3d_cov