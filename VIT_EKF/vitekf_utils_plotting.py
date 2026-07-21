"""
    Plotting Utilities
    Supplements VIT-EKF by providing utilities for matplotlib data visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import VIT_EKF.vitekf_utils_core as utils
from types import SimpleNamespace
from scipy.stats import multivariate_normal

#                            X       Z
image_extent_vertical = [-20, 20,  0, 80] # m
#                            Z       Y 
image_extent_lateral  = [  0, 80,  5, -15]  # m
# 
def ComputeGaussianCrossSectionalViews(tracks):
    """Compute Gaussian Cross Sectional Views
    Args:
        tracks: 
    Returns:
        gaussian_xsection Struct with Lateral & Vertical Images
    """
    num_points = utils.AssertTracksShape(tracks)
    image_size = 300  # pixels
    # Vertical View
    x = np.linspace(image_extent_vertical[0], image_extent_vertical[1], image_size)
    y = np.linspace(image_extent_vertical[2], image_extent_vertical[3], image_size)
    X, Y = np.meshgrid(x, y)
    pos_vert = np.dstack((X, Y))  # Stack grid arrays for multivariate normal evaluation
    # Lateral View
    x = np.linspace(image_extent_lateral[0], image_extent_lateral[1], image_size)
    y = np.linspace(image_extent_lateral[2], image_extent_lateral[3], image_size)
    X, Y = np.meshgrid(x, y)
    pos_lat = np.dstack((X, Y))  # Stack grid arrays for multivariate normal evaluation
    # 
    gaussian_xsection_vert = np.zeros((image_size, image_size))
    gaussian_xsection_lat  = np.zeros((image_size, image_size))
    for i_point in range(num_points):
        # Create the multivariate Gaussian distribution
        mean_xz = tracks.pts_3d_mean[(0,2),i_point]
        cov = tracks.pts_3d_cov[i_point,:,:] 
        # print('cov')
        # print(cov)
        cov_xz = cov[np.ix_([0,2],[0,2])]
        # print('cov_xz', cov_xz)
        # print('cov_xz.shape', cov_xz.shape)
        gaussian = multivariate_normal(mean=mean_xz, 
                                       cov=cov_xz, allow_singular=True)        
        z = gaussian.pdf(pos_vert)
        gaussian_xsection_vert += z
        # 
        mean_zy = tracks.pts_3d_mean[(2,1),i_point]
        cov_zy = cov[np.ix_([2,1],[2,1])]
        gaussian = multivariate_normal(mean=mean_zy, 
                                       cov=cov_zy, allow_singular=True)        
        z = gaussian.pdf(pos_lat)
        gaussian_xsection_lat += z
    # 
    gaussian_xsection = SimpleNamespace()
    gaussian_xsection.vert = gaussian_xsection_vert
    gaussian_xsection.lat  = gaussian_xsection_lat

    return gaussian_xsection

def InitializeVisualization(images, tracks):
    """Initialize Visualization
    Args:
        images:
        tracks: 
    Returns:
        plot_handles:
    """
    num_points = utils.AssertTracksShape(tracks)
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # Init Visualization
    layout = [
        ['left1', 'left1', 'middle1', 'right1'],
        ['left1', 'left1', 'middle1', 'right1'],
        ['left1', 'left1', 'middle1', 'right1'],
        ['left2', 'left2', 'middle2', 'right2']
    ]
    fig, axd = plt.subplot_mosaic(layout, layout="constrained")
    fig.set_size_inches((30,15))
    fig.suptitle('(Semi) Monocular Visual-Inertial Tracking with EKF     ', 
                 fontsize=60, weight='bold')
    title_font_size = 32
    label_font_size = 25
    # Access and plot on individual axes using their text keys
    # Left 1 - Left Image with Tracked 2D Features & Reprojected 3D Points
    img_left_imshow  = axd['left1'].imshow(images.left, cmap='gray', vmin=0, vmax=255)
    pts_left_scatter = axd['left1'].scatter(tracks.pts_2d_left[0,:], 
                                            tracks.pts_2d_left[1,:], 
                                            color='cyan', s=80, label='Tracked Feature')
    pts_left_reproj  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)[:-1,:]
    pts_left_reproj_scatter = axd['left1'].scatter(pts_left_reproj[0,:], 
                                                   pts_left_reproj[1,:], 
                                                   edgecolors='green', facecolors='none', linewidths=3, s=80, 
                                                   label='Corrected Reproj')
    reproj_prefit_scatter = axd['left1'].scatter(pts_left_reproj[0,:], 
                                                 pts_left_reproj[1,:], 
                                                 edgecolors='red', facecolors='none', linewidths=1.5, s=80, 
                                                 label='Predicted Reproj')
    axd['left1'].legend(loc='upper right', fontsize=16)    
    axd['left1'].set_title('Image with Tracked 2D Features & 3D Estimates Reprojected', 
                           fontsize=title_font_size, weight='bold')    
    axd['left1'].set_xticks([])
    axd['left1'].set_yticks([])
    # Left 2
    # Reprojection Errors (Residuals)
    post_fit_res = np.sum((tracks.pts_2d_left - pts_left_reproj)**2, axis=0)**(0.5)
    post_fit_res_plot  = axd['left2'].plot(np.arange(num_points), post_fit_res, 
                                           color='green', label='Measurement Corrected', linewidth=4.0)
    pre_fit_res_plot   = axd['left2'].plot(np.arange(num_points), post_fit_res, 
                                           color='red',  label='Predicted', linewidth=3.0)
    axd['left2'].set_title('Residuals: Reprojection Errors', fontsize=title_font_size, weight='bold')
    axd['left2'].set_xlabel('Track Index', fontsize=label_font_size, weight='bold')
    axd['left2'].set_ylabel('Distance (pixels)', fontsize=label_font_size, weight='bold')
    axd['left2'].legend(loc='upper right', fontsize=18)  
    axd['left2'].grid(True)
    axd['left2'].set_ylim(0,40)
    # ----------------------------------------------------------------------------------------
    # Middle 1 PREDICT
    vert_pred_scatter = axd['middle1'].scatter(tracks.pts_3d_mean[0,:], 
                                                 tracks.pts_3d_mean[2,:], 
                                                 facecolors='violet',   
                                                 edgecolors='black',   
                                                 linewidths=2.5,        
                                                 s=100)
    vert_pred_imshow  = axd['middle1'].imshow(gaussian_xsection.vert, 
                                                cmap='magma', origin='lower', extent=image_extent_vertical, vmin=0, vmax=0.07)
    axd['middle1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size, weight='bold')
    axd['middle1'].set_ylabel('Vertical (Birds Eye) View)', fontsize=title_font_size, weight='bold')
    axd['middle1'].set_title('Predicted', fontsize=title_font_size+5, weight='bold')
    axd['middle1'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['middle1'].set_aspect("equal")
    # Middle 2 PREDICT
    lateral_pred_scatter = axd['middle2'].scatter(tracks.pts_3d_mean[2,:], 
                                              tracks.pts_3d_mean[1,:], 
                                              facecolors='violet',   
                                              edgecolors='black',   
                                              linewidths=2.5,        
                                              s=100)                 
    lateral_pred_imshow = axd['middle2'].imshow(gaussian_xsection.lat, 
                                            cmap='magma', origin='lower', extent=image_extent_lateral, vmin=0, vmax=0.07)
    axd['middle2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['middle2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['middle2'].set_ylabel('Lateral (Side) View', fontsize=title_font_size, weight='bold')
    axd['middle2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    # ----------------------------------------------------------------------------------------
    # Right 1 CORRECT
    vert_corr_scatter = axd['right1'].scatter(tracks.pts_3d_mean[0,:], 
                                              tracks.pts_3d_mean[2,:], 
                                              facecolors='violet',   
                                              edgecolors='black',   
                                              linewidths=2.5,        
                                              s=100)
    vert_corr_imshow  = axd['right1'].imshow(gaussian_xsection.vert, 
                                             cmap='magma', origin='lower', extent=image_extent_vertical, vmin=0, vmax=0.07)
    axd['right1'].set_xlabel('X Lateral Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_ylabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right1'].set_title('Measurement Corrected', fontsize=title_font_size+5, weight='bold')
    axd['right1'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    axd['right1'].set_aspect("equal")
    # Right 2 CORRECT
    lateral_corr_scatter = axd['right2'].scatter(tracks.pts_3d_mean[2,:], 
                                                 tracks.pts_3d_mean[1,:], 
                                                 facecolors='violet',   
                                                 edgecolors='black',   
                                                 linewidths=2.5,        
                                                 s=100)                 
    lateral_corr_imshow = axd['right2'].imshow(gaussian_xsection.lat, 
                                               cmap='magma', origin='lower', extent=image_extent_lateral, vmin=0, vmax=0.07)
    axd['right2'].set_ylim((image_extent_lateral[2], image_extent_lateral[3]))
    axd['right2'].set_xlabel('Z Longitudinal Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].set_ylabel('Y Vertical Axis (m)', fontsize=label_font_size, weight='bold')
    axd['right2'].grid(True, color='violet', linestyle='--', linewidth=0.5)
    ## ===================================================================================
    plot_handles = SimpleNamespace()
    plot_handles.fig = fig
    plot_handles.axd = axd
    # Left 1
    plot_handles.img_left_imshow = img_left_imshow
    plot_handles.pts_left_scatter = pts_left_scatter
    plot_handles.pts_left_reproj_scatter = pts_left_reproj_scatter
    plot_handles.reproj_prefit_scatter = reproj_prefit_scatter
    # Left 2 
    plot_handles.post_fit_res_plot  = post_fit_res_plot 
    plot_handles.pre_fit_res_plot   = pre_fit_res_plot    
    # Middle 1
    plot_handles.vert_pred_scatter = vert_pred_scatter
    plot_handles.vert_pred_imshow = vert_pred_imshow                      
    # Middle 2
    plot_handles.lateral_pred_scatter = lateral_pred_scatter
    plot_handles.lateral_pred_imshow = lateral_pred_imshow
    # Right 1
    plot_handles.vert_corr_scatter = vert_corr_scatter
    plot_handles.vert_corr_imshow = vert_corr_imshow                      
    # Right 2
    plot_handles.lateral_corr_scatter = lateral_corr_scatter
    plot_handles.lateral_corr_imshow = lateral_corr_imshow
    
    return plot_handles
    
def UpdatePredictedXSectionViews(plot_handles,
                                 tracks):
    """Update Prediction Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # axd['middle2']
    plot_handles.lateral_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)
    plot_handles.lateral_pred_imshow.set_data(gaussian_xsection.lat) # Y-Axis Points Downward
    # axd['middle1']
    plot_handles.vert_pred_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    plot_handles.vert_pred_imshow.set_data(gaussian_xsection.vert)

    return

def UpdateCorrectedXSectionViews(plot_handles,
                                 tracks):
    """Update Corrected Tracks Cross-Sectional Views
    Args:
        plot_handles:
        tracks: 
    Returns:
    """
    gaussian_xsection = ComputeGaussianCrossSectionalViews(tracks)
    # axd['middle2']
    plot_handles.lateral_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[2,:], tracks.pts_3d_mean[1,:])).T)
    plot_handles.lateral_corr_imshow.set_data(gaussian_xsection.lat) # Y-Axis Points Downward
    # axd['middle1']
    plot_handles.vert_corr_scatter.set_offsets(np.vstack((tracks.pts_3d_mean[0,:], tracks.pts_3d_mean[2,:])).T)
    plot_handles.vert_corr_imshow.set_data(gaussian_xsection.vert)

    return

def UpdateImageAndReprojectionError(plot_handles, 
                                    images_next, 
                                    tracks,
                                    reproj_pred):
    """Update Image, Reprojected Points, and Reprojection Error Plots
    Args:
        plot_handles:
        images_next:
        tracks: 
    Returns:
    """
    num_points = utils.AssertTracksShape(tracks)
    print('reproj_pred.shape ', reproj_pred.shape)
    print('num_points ', num_points)
    assert reproj_pred.shape == (2, num_points)
    x_values = np.arange(num_points)
    # Optimal Tracked 3D Points Reprojected into Cameras
    pts_left_reproj  = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)[:-1,:]  # Only Left is used in KF
    post_fit_res = np.sum((tracks.pts_2d_left - pts_left_reproj)**2, axis=0)**(0.5)
    pre_fit_res  = np.sum((tracks.pts_2d_left - reproj_pred)**2, axis=0)**(0.5)
    # Left 1
    plot_handles.img_left_imshow.set_data(images_next.left)
    plot_handles.pts_left_scatter.set_offsets(tracks.pts_2d_left.T)
    plot_handles.pts_left_reproj_scatter.set_offsets(pts_left_reproj.T)
    plot_handles.reproj_prefit_scatter.set_offsets(reproj_pred.T)
    # Left 2  
    plot_handles.post_fit_res_plot[0].set_xdata(x_values) 
    plot_handles.post_fit_res_plot[0].set_ydata(post_fit_res)
    plot_handles.pre_fit_res_plot[0].set_xdata(x_values) 
    plot_handles.pre_fit_res_plot[0].set_ydata(pre_fit_res)
    plot_handles.axd['left2'].set_xlim(0, num_points)     

    return