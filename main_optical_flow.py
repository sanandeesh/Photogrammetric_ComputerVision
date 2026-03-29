import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

"""
This script demonstrates Optical Flow (Ma et al, 2004) as
    - Multi-Scale (Pyramid) 
    - Gradient based (Lukas Kanade)  
    ... over KITTI images.

Note: Significant noise in the Optical-Flow vertical/y component needs to be investigated (see YT videos).
The noise is highly correlated across the whole image/grid. The OF horizontal/x component looks correct.
-------------------------------------

Ma Y, Soatto S, Kosecká, J, & Sastry S S (2004). 
_An Invitation to 3-D Vision: From Images to Geometric Models_. 
Springer-Verlag.

Torralba, A. and Isola, P. and Freeman, W.T. 
_Foundations of Computer Vision_, 
https://visionbook.mit.edu/optical_flow.html
"""

# Constants
kNumOFGridRows = 12
kNumOFGridColumns = 40
kWindowHalfLength = 15
kImageShape = [375, 1242] 
kMinFeatureThresh = 20 # Harris Criterion Threshold (Edge Reliability)

def GetLeftImage(data_path, index):
    """Return Left KITTI Cam-Image at Index

    Args:
        index: Index of image (proxy for timestamp)

    Returns:
        Left Image at Index.
    """
    width = 10
    padded_number = str(index).zfill(width)
    print(padded_number)
    left_image_name  = data_path + 'image_00/data/' + padded_number + '.png'
    # 
    left_image_orig = plt.imread(left_image_name)
    #
    return left_image_orig

def ComputeOFGrid(img_shape):
    """Compute Fixed Mesh Grid for Optical Flow locations

    Args:
        img_shape: Image dimensions against which OF Grid is created

    Returns:
        Optical Flow Grid for Quiver-Plot
    """
    rows = np.linspace(0, img_shape[0], kNumOFGridRows)
    rows = rows[1:-1]
    rows = np.round(rows).astype(int)
    cols = np.linspace(0, img_shape[1], kNumOFGridColumns)
    cols = cols[1:-1]
    cols = np.round(cols).astype(int)
    mesh_x, mesh_y = np.meshgrid(cols, rows)
    # 
    return rows, cols, mesh_x, mesh_y

def GetSobelGradients(image):
    """Compute Sobel (Horizontal & Vertical) Gradients

    Args:
        image: input image

    Returns:
        Vertical (X-direction), and Horizontal (Y-direction) Gradients
    """
    sobel_h = ndimage.sobel(image, 0)  # horizontal (Y) gradient 
    sobel_v = ndimage.sobel(image, 1)  # vertical (X) gradient 

    return sobel_v, sobel_h

def ComputeOpticalFlow(gradient_x, gradient_y, gradient_t, mesh_x, mesh_y):
    """Compute Optical Flow given input Gradients

    Args:
        gradient_x: Vertical gradient image
        gradient_y: Horizontal gradient image
        gradient_t: Temporal gradient image
        mesh_x: Optical Flow Grid x-indexes
        mesh_y: Optical Flow Grid y-indexes

    Returns:
        d_x: Optical Flow Grid x-values
        d_y: Optical Flow Grid y-values
        C: Optical Flow Grid - Harris-Criterion edge reliability
    """
    assert mesh_x.shape == mesh_y.shape
    assert (gradient_x.shape == gradient_y.shape) and (gradient_x.shape == gradient_t.shape)

    d_x, d_y, C = np.zeros(mesh_x.shape), np.zeros(mesh_x.shape), np.zeros(mesh_x.shape)
    k = kWindowHalfLength
    for i_row in np.arange(mesh_x.shape[0]):
        for i_col in np.arange(mesh_x.shape[1]):
            # Gradient Matrix
            x, y = mesh_x[i_row, i_col], mesh_y[i_row, i_col]
            #
            grad_x = gradient_x[y-k:y+k, x-k:x+k]
            grad_y = gradient_y[y-k:y+k, x-k:x+k]
            grad_t = gradient_t[y-k:y+k, x-k:x+k]
            G = np.zeros((2,2))
            G[0,0] = np.sum(grad_x*grad_x)
            G[1,1] = np.sum(grad_y*grad_y)
            G[0,1] = np.sum(grad_x*grad_y)
            G[1,0] = np.sum(grad_y*grad_x)
            # B
            b = np.zeros((2,1))
            b[0] = np.sum(grad_x * grad_t)
            b[1] = np.sum(grad_y * grad_t)
            # Harris Criterion (Edge Reliability)
            k_quality = 0.05
            q = np.linalg.det(G) - (k_quality * np.trace(G)**2) # quality
            if q > kMinFeatureThresh:
                # ----------------------------------------------------------
                # Ma et al, Ch 4.3.1 "Feature Tracking & Optical Flow" 
                # Least Squares Estimate of Image Velocity fulfilling "Brightness Constancy Constraint"
                d = -np.linalg.inv(G) @ b
                # ----------------------------------------------------------
            else: # Edge not strong enough, inversion is not advisable
                d = np.zeros((2,1))
            d_x[i_row, i_col] = d[0,0]
            d_y[i_row, i_col] = d[1,0]
            C[i_row, i_col] = q

    return d_x, d_y, C

def ComputeOpticalFlowFromImagePair(image, image_next, mesh_x, mesh_y):
    """Compute Optical Flow given input Image pair

    Args:
        image: current image
        image_next: next image
        mesh_x: Optical Flow Grid x-indexes
        mesh_y: Optical Flow Grid y-indexes

    Returns:
        d_x: Optical Flow Grid x-values
        d_y: Optical Flow Grid y-values
        C: Optical Flow Grid - Edge reliability quantity
    """
    gradient_x, gradient_y = GetSobelGradients(image)
    gradient_t = (image_next - image)
    d_x, d_y, C = ComputeOpticalFlow(gradient_x, gradient_y, gradient_t, mesh_x, mesh_y)    
    
    return d_x, d_y, C

def DownSampleImg(image):
    """Smooth, and Down sample input image (to half size)

    Args:
        image:

    Returns:
        image_dwnsmpl: downsampled image
    """
    image = ndimage.gaussian_filter(image, sigma=1)
    image_dwnsmpl = image[0::2, 0::2]

    return image_dwnsmpl


if __name__ == "__main__":
    data_path = '2011_09_26/2011_09_26_drive_0001_sync/'
    # Image Pyramid Dimensions
    img_shape_0 = np.round(kImageShape).astype(int)
    rows_0, cols_0, mesh_x_0, mesh_y_0 = ComputeOFGrid(img_shape_0)
    img_shape_1 = np.round(img_shape_0/2).astype(int)
    rows_1, cols_1, mesh_x_1, mesh_y_1 = ComputeOFGrid(img_shape_1)
    img_shape_2 = np.round(img_shape_1/2).astype(int)
    rows_2, cols_2, mesh_x_2, mesh_y_2 = ComputeOFGrid(img_shape_2)
    img_shape_3 = np.round(img_shape_2/2).astype(int)
    rows_3, cols_3, mesh_x_3, mesh_y_3 = ComputeOFGrid(img_shape_3)
    # Read Image-Pair, recusrively downsample a pyramid
    image_0 = GetLeftImage(data_path, 0)
    image_next_0 = GetLeftImage(data_path, 1)
    image_1, image_next_1 = DownSampleImg(image_0), DownSampleImg(image_next_0)
    image_2, image_next_2 = DownSampleImg(image_1), DownSampleImg(image_next_1)
    image_3, image_next_3 = DownSampleImg(image_2), DownSampleImg(image_next_2)
    # ---------------------------------------------
    # Optical Flow Computation (over Mesh-Grid)
    # ---------------------------------------------
    d_x_0, d_y_0, C_0 = ComputeOpticalFlowFromImagePair(image_0, image_next_0, mesh_x_0, mesh_y_0)
    d_x_1, d_y_1, C_1 = ComputeOpticalFlowFromImagePair(image_1, image_next_1, mesh_x_1, mesh_y_1)
    d_x_2, d_y_2, C_2 = ComputeOpticalFlowFromImagePair(image_2, image_next_2, mesh_x_2, mesh_y_2)
    d_x_3, d_y_3, C_3 = ComputeOpticalFlowFromImagePair(image_3, image_next_3, mesh_x_3, mesh_y_3)
    # ---------------------------------------------
    # Visualization
    # ---------------------------------------------
    quiver_scale = 0.02
    plt.figure(figsize=(30, 18))
    plt.suptitle('\nMulti-Scale Gradient based Optical Flow', fontsize=40)
    # Level 0 (Original)
    plt.subplot(4, 1, 1)
    plt.title(str('Original Level 0 ' + str(img_shape_0)), fontsize=25)
    plt.axis('off')
    imshow_0 = plt.imshow(image_0, cmap='gray', vmin=0, vmax=1)
    of_grid_0 = plt.quiver(cols_0, 
                           rows_0, 
                           d_x_0 + 2*d_x_1 + 4*d_x_2 + 8*d_x_3, 
                           d_y_0 + 2*d_y_1 + 4*d_y_2 + 8*d_y_3, 
                           C_0 + C_1 + C_2 + C_3,
                           angles='xy', scale_units='xy', scale=quiver_scale, cmap='cool')
    # Level 1 
    plt.subplot(4, 1, 2)
    plt.title(str('Level 1 ' + str(img_shape_1)), fontsize=25)
    plt.axis('off')
    imshow_1 = plt.imshow(image_1, cmap='gray', vmin=0, vmax=1)
    of_grid_1 = plt.quiver(cols_1, 
                           rows_1, 
                           d_x_1 + 2*d_x_2 + 4*d_x_3, 
                           d_y_1 + 2*d_y_2 + 4*d_y_3, 
                           C_1 + C_2 + C_3,
                           angles='xy', scale_units='xy', scale=quiver_scale, cmap='cool')
    # Level 2
    plt.subplot(4, 1, 3)
    plt.title(str('Level 2 ' + str(img_shape_2)), fontsize=25)
    plt.axis('off')
    imshow_2 = plt.imshow(image_2, cmap='gray', vmin=0, vmax=1)
    of_grid_2 = plt.quiver(cols_2, 
                           rows_2, 
                           d_x_2 + 2*d_x_3, 
                           d_y_2 + 2*d_y_3, 
                           C_2 + C_3,
                           angles='xy', scale_units='xy', scale=quiver_scale, cmap='cool')
    # Level 3
    plt.subplot(4, 1, 4)
    plt.title(str('Level 3 ' + str(img_shape_3)), fontsize=25)
    plt.axis('off')
    imshow_3 = plt.imshow(image_3, cmap='gray', vmin=0, vmax=1)
    of_grid_3 = plt.quiver(cols_3, 
                           rows_3, 
                           d_x_3, 
                           d_y_3,
                           C_3,
                           angles='xy', scale_units='xy', scale=quiver_scale, cmap='cool')

    plt.show()