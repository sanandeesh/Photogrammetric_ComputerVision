"""
        (Semi) Monocular Visual Inertial Tracking with EKF

Usage: python3 main_visual_inertial_tracking_EKF.py
Requirements: Numpy, Matplotlib, scipy, cv2, skimage, sympy
"""

import VIT_EKF.vitekf_core       as core
import VIT_EKF.vitekf_utils_core as utils
import VIT_EKF.vitekf_utils_oxts as utils_oxts
import VIT_EKF.vitekf_utils_plotting as utils_plt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_path = '2011_09_26/2011_09_26_drive_0001_sync/'
    oxts_data = utils_oxts.GetOXTSDataFull(data_path)
    images    = utils.GetImagePair(data_path, index = 0)
    tracks    = core.InitializeTracks(images)
    ## 
    plt_hndls = utils_plt.InitializeVisualization(images, tracks)
    plt.draw()
    plt.pause(3.0)
    for index in range(1, oxts_data.num_samples):
        # ------------------ I. Prediction Step ------------------  
        tracks = core.ForwardPropagate3DPoints(index, oxts_data, tracks)
        # Update Plots
        utils_plt.UpdatePredictedXSectionViews(plt_hndls, tracks)
        # ------------------ II. Track 2d-Features in Next Image\Measurement ------------------ 
        images_next = utils.GetImagePair(data_path, index)
        # Update Plots
        [images, # Updated For Next Cycle
         tracks  # Updated For Next Cycle
                      ] =  core.TrackFeaturesAndReplaceLostPoints(images, # From Last Cycle
                                                                  tracks, # From Last Cycle
                                                                  images_next)
        pts_2d_predicted_reproj = utils.ProjectPoints(tracks.pts_3d_mean, utils.P_Left)[:-1,:]
        # ------------------- III. Measurement Correction Step -------------------
        tracks = core.KFMeasurementCorrectionStep(tracks)
        # Update Plots
        utils_plt.UpdateImageAndReprojectionError(plt_hndls, 
                                                  images_next, 
                                                  tracks,
                                                  pts_2d_predicted_reproj)
        utils_plt.UpdateCorrectedXSectionViews(plt_hndls, tracks)
        plt.draw()
        plt.pause(0.1)

    plt.show()