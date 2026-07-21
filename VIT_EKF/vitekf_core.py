"""
    VIT-EKF Core
    Core VIT-EKF Functions for 
    (1) Initializing Tracks, 
    (2) Updating Tracks, 
    (3) KF Prediction-Step, 
    (4) KF Measurement-Correction Step
"""
import numpy as np
import cv2
import VIT_EKF.vitekf_utils_core as utils
from types import SimpleNamespace


def InitializeTracks(images):
    """ Initialize Tracks given Image-Pair. Tracks consist of 
    1. 2D Features Tracked
    2. 3D Triangulated Features Tracked
    3. 3x3 Covariance Matrix of each Tracked Point
    Args:
        images: Left & Right Image
    Returns:
        tracks Struct
    """
    # [N x 1 x 2]
    pts_2d_left_cv = cv2.goodFeaturesToTrack(images.left, 
                                             mask = utils.GetFeatureROIMask(images.left), 
                                             **utils.feature_params) 
    pts_2d_left = np.squeeze(pts_2d_left_cv, axis=1).T # 2xN
    pts_2d_right = utils.ComputePointCorrespondences(images.left, 
                                                     images.right, 
                                                     pts_2d_left.astype(int))
    pts_3d_mean, pts_2d_left = utils.TriangulateFeaturesRemoveInvalidPoints(pts_2d_left, 
                                                                            pts_2d_right)
    pts_3d_cov = utils.ComputeDistributionsOf3DPoints(pts_3d_mean)
    # TRACKS INITIALIZATION
    tracks = SimpleNamespace()
    tracks.pts_2d_left = pts_2d_left # N Observations [2,N]
    tracks.pts_3d_mean = pts_3d_mean # N Exp States   [4,N] (inc Homog Coord = 1.0)
    tracks.pts_3d_cov  = pts_3d_cov  # N Covariances  [N,3,3] 
    #
    num_points = utils.AssertTracksShape(tracks)

    return tracks

def TrackFeaturesAndReplaceLostPoints(images,
                                      tracks,
                                      images_next):
    """ 
    Track features across two images, and replace lost points (e.g. left FoV)
    Args:
        images: Left & Right Image
        tracks: 
        images_next: images of next time-step
    Returns:
        images: Updated to `images_next`
        tracks: Updated
    """
    pts_2d_left_cv = tracks.pts_2d_left.T.reshape(-1,1,2) # [N,1,2]
    pts_3d_mean    = tracks.pts_3d_mean                   # [4,N]  (Already Forward-Propagated!)
    pts_3d_cov     = tracks.pts_3d_cov                    # [3,3,N] 
    pts_2d_left_next_cv, st, err = cv2.calcOpticalFlowPyrLK(images.left, 
                                                            images_next.left, 
                                                            pts_2d_left_cv, 
                                                            None, 
                                                            **utils.lk_params)
    # Remove Invalid Tracks (e.g. left FoV)
    pts_2d_left_next = pts_2d_left_next_cv[st==1].T          # [2,N]
    pts_3d_mean_next = pts_3d_mean[:,(st.T == 1).squeeze()]  # [4,N]
    pts_3d_cov_next  = pts_3d_cov[(st.T == 1).squeeze(),:,:] # [3,3,N]
    # Replace points lost outside the FoV
    num_lost_points = tracks.pts_2d_left.shape[1] - pts_2d_left_next.shape[1]
    assert (num_lost_points == (pts_3d_mean.shape[1] - pts_3d_mean_next.shape[1])) 
    assert (pts_3d_mean_next.shape[1] == pts_2d_left_next.shape[1])
    if num_lost_points:
        utils.feature_params['maxCorners'] = num_lost_points
        pts_2d_left_new_cv = cv2.goodFeaturesToTrack(images_next.left, 
                                                     mask = utils.GetFeatureROIMask(images_next.left), 
                                                     **utils.feature_params)
        pts_2d_left_new  = np.squeeze(pts_2d_left_new_cv, axis=1).T
        # 
        pts_2d_right_new = utils.ComputePointCorrespondences(images_next.left, 
                                                       images_next.right, 
                                                       pts_2d_left_new.astype(int))
        pts_3d_mean_new, pts_2d_left_new = utils.TriangulateFeaturesRemoveInvalidPoints(pts_2d_left_new, 
                                                                                         pts_2d_right_new, 
                                                                                         0.0005)
        pts_3d_cov_new = utils.ComputeDistributionsOf3DPoints(pts_3d_mean_new)
        pts_2d_left_next = np.concatenate((pts_2d_left_next, pts_2d_left_new), axis=1) # Horiz Concat
        pts_3d_mean_next = np.concatenate((pts_3d_mean_next, pts_3d_mean_new), axis=1) # Horiz Concat
        pts_3d_cov_next  = np.concatenate((pts_3d_cov_next, pts_3d_cov_new),   axis=0) # Stack Concat
        assert ((pts_3d_mean_next.shape[1] == pts_2d_left_next.shape[1]) and 
                (pts_3d_mean_next.shape[1] == pts_3d_cov_next.shape[0]))

    # Now update the previous frame and previous points
    images.left  = images_next.left.copy()
    images.right = images_next.right.copy()
    tracks.pts_2d_left = pts_2d_left_next
    tracks.pts_3d_mean = pts_3d_mean_next
    tracks.pts_3d_cov  = pts_3d_cov_next

    return images, tracks

# Points Move in Ego-Frame of Moving Car, so apply the "Opposite" motion to Points
def ForwardPropagate3DPoints(index, oxts_data, tracks):
    """ Prediction Step: apply Forward\Lateral translation, and Yaw\Pitch rotation from KITTI OXTS to 3D Tracks.
    Args:
        index: timestep index to grab corresponding OXTS inertial data
        oxts_data: 
        tracks:
    Returns:
        Forward Propagated `tracks`
    """
    num_points = utils.AssertTracksShape(tracks)
    delta_seconds = oxts_data.local_timestamps[index] - oxts_data.local_timestamps[index-1]
    v_forward =  (oxts_data.vf[index-1] + oxts_data.vf[index])/2.0 
    v_right   = -(oxts_data.vl[index-1] + oxts_data.vl[index])/2.0 # Switch from Leftward to Rightward Velocity
    w_up      =  (oxts_data.wu[index-1] + oxts_data.wu[index])/2.0 # Angular Velocity about vertical Axis (CCW)
    w_left    =  (oxts_data.wl[index-1] + oxts_data.wl[index])/2.0 # Angular Velocity about Left Axis (?)
    delta_forward = v_forward*delta_seconds
    delta_right   = v_right*delta_seconds
    delta_yaw     = w_up*delta_seconds    # CounterClockwise
    delta_pitch   = w_left*delta_seconds    # ?
    #
    R_y = np.array([[np.cos(delta_yaw), 0.0,  np.sin(delta_yaw)],   # Clockwise
                    [             0.0,  1.0,                0.0],   
                    [-np.sin(delta_yaw), 0.0, np.cos(delta_yaw)]]) 
    R_x = np.array([[1.0,                 0.0,                   0.0],
                    [0.0, np.cos(delta_pitch), -np.sin(delta_pitch)],
                    [0.0, np.sin(delta_pitch),  np.cos(delta_pitch)]])    
    shift = np.array([[-delta_right],   # X Axis Rightward +
                               [0.0],   # Y Axis Downward +
                     [-delta_forward]]) # Z Axis Forward +
    # Predict Mean
    pts_3d_mean_pred = R_x @ R_y @ (tracks.pts_3d_mean[:-1,:] + shift) # Ignore the Homogenous Coordinates
    pts_3d_mean_pred = np.concat((pts_3d_mean_pred, np.ones((1,num_points))), axis=0)
    # Predict Covariance
    Q_input_space = utils.ComputeOdometryCovariance(v_forward,
                                              v_right,
                                              w_up,
                                              w_left,
                                              delta_seconds)
    Q_state_space = np.zeros(tracks.pts_3d_cov.shape)
    for i_point in range(num_points):
        Q_state_space[i_point,:,:] = utils.ComputeProcessNoise(Q_input_space, 
                                                               tracks.pts_3d_mean[:-1,i_point],
                                                               delta_forward,
                                                               delta_right)
    Q = Q_state_space
    pts_3d_cov_pred = R_y @ tracks.pts_3d_cov @ R_y.T + Q 
    # Update Tracks 
    tracks.pts_3d_mean = pts_3d_mean_pred
    tracks.pts_3d_cov  = pts_3d_cov_pred

    return tracks


"""
See `vitekf_compute_jacobians.py` for details
--------------------------------------------------------------------------------------------------------------------
------------------------------------------ Measurement Model Jacobian ---------------------------------------------- 
Measurement Model: Non-Linear Normalized Camera Projection
     ⎡fₓ        cₓ ⎤      
     ⎢───  0.0  ───⎥ ⎡μₓ ⎤
     ⎢μ_z       μ_z⎥ ⎢   ⎥
 h = ⎢             ⎥⋅⎢μ_y⎥
     ⎢     f_y  c_y⎥ ⎢   ⎥
     ⎢0.0  ───  ───⎥ ⎣μ_z⎦
     ⎣     μ_z  μ_z⎦      
--------------------------------------------------
Jacobian wrt State (μ_x μ_y,μ_z):
     ⎡fₓ         -fₓ⋅μₓ  ⎤
     ⎢───   0    ─────── ⎥
     ⎢μ_z            2   ⎥
     ⎢            μ_z    ⎥
 H = ⎢                   ⎥
     ⎢     f_y  -f_y⋅μ_y ⎥
     ⎢ 0   ───  ─────────⎥
     ⎢     μ_z       2   ⎥
     ⎣            μ_z    ⎦
--------------------------------------------------------------------------------------------------------------------
"""
def ComputeMeasurementModelJacobian(mean_predicted):
    """ Compute Measurement Model Jacobian
    Args:
        mean_predicted: 

    Returns:
        Jacobian of Non-Linear Projection Model of Left Camera
    """
    assert (mean_predicted.shape == (3,1))
    X = mean_predicted[0,0] 
    Y = mean_predicted[1,0] 
    Z = mean_predicted[2,0] 
    fx = utils.P_Left[0,0]
    fy = utils.P_Left[1,1]
    H = np.array([
        [fx / Z,   0.0,    -(fx * X) / (Z**2)],
        [0.0,    fy / Z,   -(fy * Y) / (Z**2)]
    ])

    return H

def KFMeasurementCorrectionStep(tracks):
    """ Measurement Correction Step: Update Predicted Tracks with Measurement Error time Kalman Gain
    Args:
        tracks: 
    Returns:
        Updated tracks
    """    
    num_points = utils.AssertTracksShape(tracks)
    pts_reproj_left  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)[:-1,:]  # Only Left is used in KF
    assert (pts_reproj_left.shape == (2, num_points))
    measurement_error = tracks.pts_2d_left - pts_reproj_left
    R = 5*np.eye(2) # Measurement Covariance (2,2)
    for i_point in range(num_points):
        mean_predicted = tracks.pts_3d_mean[:-1,i_point].reshape(3,1)
        Z = mean_predicted[2]
        cov_pred = tracks.pts_3d_cov[i_point,:,:].copy()
        H = ComputeMeasurementModelJacobian(mean_predicted)
        S = (H @ cov_pred @ H.T) + R
        S_inv = np.linalg.inv(S)
        KalmanGain = (cov_pred @ H.T) @ S_inv
        mean_corrected = mean_predicted + KalmanGain @ measurement_error[:,i_point].reshape(2,1) # residual
        i_minus_KH = np.eye(3) - KalmanGain @ H
        # Joseph Form
        cov_corrected = (i_minus_KH @ cov_pred @ i_minus_KH.T) + (KalmanGain @ R @ KalmanGain.T)
        # cov_corrected = i_minus_KH @ cov_pred
        #
        tracks.pts_3d_mean[:,i_point] = np.vstack((mean_corrected, 1.0)).reshape((4,))
        tracks.pts_3d_cov[i_point,:,:] = cov_corrected

    return tracks