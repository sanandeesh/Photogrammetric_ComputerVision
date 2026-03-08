import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage # Sobel Filtering
import skimage as ski # Template Matching

sobel_threshold = 1.5 # Edge-Feature Threshold 
kernel_half_length = 7 # Template-Matching Window Half-Size (Pixels)

"""
This script demonstrates DLT Triangulation (described by MVG) of calibrated stereo camera data provided by the "KITTI Dataset".
The two cameras are laterally seperated by 0.54m, but otherwise have identical exterior/interior orientation.
The world origin is the Left Camera centre. (X-Axis rightward, Y-Axis downward, Z-Axis forward)
The Pixel-space origin is top-left corner of the image.
The DLT-Triangulation algorithms is:
1. Features points (strong edges) are extracted from the Left Camera.
2. Corresponding features are found on the Right Camera.
3. DLT Triangulate 3D homogenous coordinates from feature point-pairs and camera matrices.
Note: The Units of the output are "normalized homogenous coordinates" based on the provided calibrated Cameras.
    To verify/calibrate these units against meters, comparison against ground truth (e.g. Lidar) is needed.

-------------------------------------
Geiger A, Lenz P, Stiller C, Urtasun R, 
Vision meets Robotics: The KITTI Dataset, 
International Journal of Robotics Research (IJRR), 2013, 
https://www.cvlibs.net/datasets/kitti/raw_data.php

Hartley R, Zisserman A, 
Multiple View Geometry in Computer Vision, 
Cambridge University Press, 2003, 2nd edition
"""

# Calibrated Camera Matrixes (Stereo Pair, 0.54m apart), from 2011_09_26 recording
P_Left = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
                   [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                   [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

P_Right = np.array([[7.215377e+02, 0.000000e+00, 6.095593e+02, -3.875744e+02],
                    [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
                    [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]])

# Canonical Fundamental Matrix of Two Cameras Seperated along X-Axis (Pure Translation, No Rotation)
# MVG, Ch-9.3.1 "Fundamental Matrices arising from special motions", Example 9.6
# Given Left-Cam point x, `l = F @ x` ---> defines the line (ax + by + c = 0) along which the corresponding point is found on Right-Cam.
# However, this script exploits the vertical alignment of the two cameras, and instead scans for correspondence-features along the same row of the template-feature.
F = np.array([[0.0, 0.0,  0.0],
              [0.0, 0.0, -1.0],
              [0.0, 1.0,  0.0]])

def GetImagePair(data_path, index):
    """Return Left/Right Image Pair at Index

    Args:
        data_path: Path to particular KITTI test collection
        index: Index of image (proxy for timestamp)

    Returns:
        Left/Right Image Pair at Index.
    """
    width = 10
    padded_number = str(index).zfill(width)
    print('Reading image-pair: ', padded_number)
    left_image_name  = data_path + 'image_00/data/' + padded_number + '.png'
    right_image_name = data_path + 'image_01/data/' + padded_number + '.png'
    left_image = plt.imread(left_image_name)
    right_image = plt.imread(right_image_name)
    print('left_image.shape ', left_image.shape)
    print('right_image.shape ', right_image.shape)

    return left_image, right_image

def ExtractImageFeatures(image):
    """Extract features of interest as pixel coordinates. 

    Args:
        image: Image from which features (strong edges) are extracted

    Returns:
        Pixel coordinates of feature points
    """
    sobel_h = ndimage.sobel(image, 0)  # horizontal gradient
    sobel_v = ndimage.sobel(image, 1)  # vertical gradient
    sobel_mag = np.sqrt(sobel_h**2 + sobel_v**2)
    # Zero-Out the Borders
    border_buffer = 3*kernel_half_length
    sobel_mag[0:border_buffer, :] = 0.0
    sobel_mag[-border_buffer:, :] = 0.0
    sobel_mag[:, 0:border_buffer] = 0.0
    sobel_mag[:, -border_buffer:] = 0.0
    #
    features = np.nonzero(sobel_mag > sobel_threshold)
    num_points = len(features[0])
    # Note: If Fundamental Matrix is used, this requires 3rd homogenous coordinate `1`.
    feature_pixels = np.vstack((features[1],           # Col/X
                                features[0],           # Row/Y
                                np.ones((num_points,), dtype=int))) # Homogenous
    print('feature_pixels.shape ', feature_pixels.shape)

    return feature_pixels

def ComputePointCorrespondences(left_image, right_image, left_poi):
    """Compute feature point correspondences in right image, given left image and its feature points

    Args:
        left_image: from which original feature template/kernel is extracted
        right_image: upon which corresponding feature is found
        left_poi: pixel coordinates of original features

    Returns:
        Pixel coordinates of corresponding feature points
    """
    left_poi_rows = left_poi[1,:]
    left_poi_cols = left_poi[0,:]
    num_points = len(left_poi_rows)
    right_poi_rows = np.zeros((num_points,))
    right_poi_cols = np.zeros((num_points,))
    k = kernel_half_length
    for i in range(num_points):
        right_poi_rows[i] = left_poi_rows[i]
        kernel_rows = left_poi_rows[i] - k, left_poi_rows[i] + k
        kernel_cols = left_poi_cols[i] - k, left_poi_cols[i] + k
        kernel = left_image[kernel_rows[0]:kernel_rows[1], kernel_cols[0]:kernel_cols[1]]
        start_col = (left_poi_cols[i] - 100) if (left_poi_cols[i] > 100) else 0
        target_img_roi = right_image[kernel_rows[0]:kernel_rows[1], start_col:(left_poi_cols[i]+k)]
        # 
        correlate_result = ski.feature.match_template(target_img_roi, kernel, pad_input=True)
        right_poi_cols[i] = np.argmax(correlate_result[k,:]) + start_col
    #
    correspondence_pixels = np.vstack((right_poi_cols,        # X
                                       right_poi_rows,        # Y
                                       np.ones((num_points,), dtype=int))) # Homogenous
    print('correspondence_pixels.shape ', correspondence_pixels.shape)
    
    return correspondence_pixels

def TriangulateSpacePoints(pixels_a, 
                           P_a, 
                           pixels_b, 
                           P_b):
    """Triangulate pairs of 2D Pixel-Coordinates into 3D (Homogenous, unnormalized) Space Coordinates 
        As described in MVG Ch-12.2 "Linear Triangulation Methods"
    Args:
        pixels_a: point features of cam a
        P_a: Projection Matrix of cam a
        pixels_b: point features of cam b
        P_b: Projection Matrix of cam b

    Returns:
        Homogenous 3D Points [4xN] 
    """
    num_points = pixels_a.shape[1]
    points_3d = np.ones((4,num_points))
    for i in range(num_points):
        A = np.zeros((6,4))
        A[0,:] = pixels_a[0,i]*P_a[2,:] - P_a[0,:]
        A[1,:] = pixels_a[1,i]*P_a[2,:] - P_a[1,:]
        A[2,:] = pixels_a[0,i]*P_a[1,:] - pixels_a[1,i]*P_a[0,:]
        A[3,:] = pixels_b[0,i]*P_b[2,:] - P_b[0,:]
        A[4,:] = pixels_b[1,i]*P_b[2,:] - P_b[1,:]
        A[5,:] = pixels_b[0,i]*P_b[1,:] - pixels_b[1,i]*P_b[0,:]
        # 
        U, S, Vh = np.linalg.svd(A)
        points_3d[:,i] = Vh[-1,:] 
    print('points_3d.shape ', points_3d.shape)

    return points_3d

def NormalizeSpacePoints(points_3d):
    """1. Remove points with very small homogenous coordinates (too close to infinity).
       2. Flip signs of triangulated points appearing behind the Camera. (i.e. mirror points)
       3. Divide each coordinate element by the last homogenous coordinate. 
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates

    Returns:
        Homogenous 3D Points [4xN] ready for plotting.
    """
    # Mirror the "Negative" Points
    i_negative_points = np.nonzero(points_3d[2,:] < 0)[0]
    points_3d[:,i_negative_points] = -1*points_3d[:,i_negative_points]
    # Remove Points Too "Close to Infinity"
    i_non_ideal_points = np.nonzero((points_3d[3,:]) > 0.01)[0]
    points_3d_normlzd = points_3d[:,i_non_ideal_points]
    homogenous_coord = points_3d_normlzd[3,:]
    # Normalize by Homogenous Coordinate
    points_3d_normlzd = points_3d_normlzd / np.vstack((homogenous_coord, homogenous_coord, homogenous_coord, homogenous_coord))
    print('points_3d_normlzd.shape ', points_3d_normlzd.shape)

    return points_3d_normlzd

def ProjectPoints(points_3d, P_Cam):
    """Project 3D homogenous to 2D homogenous coordinates, and normalize homogenous coord to 1.
       
    Args:
        points_3d: [4xN] Homogenous 3D coordinates
        P_Cam: [3x4] Projection\Camera Matrix

    Returns:
        Homogenous 2D Points [3xN] ready for plotting.
    """
    points_2d = P_Cam @ points_3d
    points_2d = points_2d / points_2d[-1,:] # Normalize by dividing by Homogenous Coordinate
    print('Projected points_2d.shape ', points_2d.shape)

    return points_2d

def VisualizeResults(left_image, points_3d):
    """Visualize triangulated points in various ways
       
    Args:
        left_image: Left Image onto which feature depth is illustrated
        points_3d: [4xN] 3D homogenous coordinates

    Returns:
    """
    points3d_x = points_3d[0,:]
    points3d_y = -points_3d[1,:] # Y-Axis points downward, so flip sign for plotting.
    points3d_z = points_3d[2,:]
    left_points_reproj = ProjectPoints(points_3d, P_Left)
    cmap_str = 'plasma'
    # 
    fig, axd = plt.subplot_mosaic([['upleft', 'right'],
                                   ['lowleft', 'right']], layout='constrained')
    fig.suptitle('DLT Triangulation of Stereo-Camera Match Points', fontsize=40)
    fig.set_size_inches((30,15))
    axd['upleft'].set_title('Left Camera with Interest-Points (Right Not Shown)', fontsize=30)
    axd['upleft'].imshow(left_image, cmap='gray', vmin=0, vmax=1)
    axd['upleft'].scatter(left_points_reproj[0,:], left_points_reproj[1,:], c=points_3d[2,:], cmap=cmap_str)
    # 
    axd['right'].set_title('Vertical Birds-Eye View', fontsize=30)
    axd['right'].scatter(points3d_x, points3d_z, c=points3d_z,cmap=cmap_str)
    axd['right'].set_xlabel('Lateral X Axis', fontsize=20)
    axd['right'].set_ylabel('Line-of-Sight Z Axis', fontsize=20)
    axd['right'].set_xlim((-30,30))
    axd['right'].set_ylim((0,100))
    axd['right'].set_aspect('equal')
    #
    axd['lowleft'].set_title('Lateral Cross-Sectional View', fontsize=30)
    axd['lowleft'].scatter(points3d_z, points3d_y, c=points3d_z, cmap=cmap_str)
    axd['lowleft'].set_xlabel('Line-of-Sight Z Axis', fontsize=20)
    axd['lowleft'].set_ylabel('Vertical Y Axis', fontsize=20)
    axd['lowleft'].set_xlim((0,100))
    axd['lowleft'].set_ylim((-10,30))
    axd['lowleft'].set_aspect('equal')

    fig = plt.figure()
    fig.set_size_inches((30,15))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(points3d_x, points3d_z, points3d_y, c=points3d_z, cmap=cmap_str)
    ax.set_title('DLT Triangulation of Stereo-Camera Match Points', fontsize=20)
    ax.set_xlabel('Lateral X Axis', fontsize=15)
    ax.set_ylabel('Longitudinal Z Axis', fontsize=15)
    ax.set_zlabel('Vertical Y Axis', fontsize=15)
    plt.axis('equal')
    plt.colorbar
    # 
    plt.show()

if __name__ == "__main__":
    # A single pair of images is provided here as a sample. The full/original are downloaded from cvlibs.net link
    data_path = '2011_09_26/2011_09_26_drive_0001_sync/'
    left_image, right_image = GetImagePair(data_path, index=0)

    ## ------------------ Stereo Triangulation -----------------------
    left_feature_pixels = ExtractImageFeatures(left_image)
    right_feature_pixels = ComputePointCorrespondences(left_image, 
                                                       right_image, 
                                                       left_feature_pixels)
    points_3d = TriangulateSpacePoints(left_feature_pixels, 
                                       P_Left,
                                       right_feature_pixels,
                                       P_Right)
    points_3d_normlzd = NormalizeSpacePoints(points_3d)
    ## --------------------------------------------------------------

    VisualizeResults(left_image, points_3d_normlzd)
    