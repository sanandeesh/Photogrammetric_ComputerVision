import main_stereo_dlt_triangulation as dlt
import numpy as np
import pytest

def test_ComputePointCorrespondences_GivenLeftPointAndStereoImagePair_ExpectCorrespondingRightPoint():
    """ Test ComputePointCorrespondences()
        Place a point in two images on the same row, but at seperate columns.
        Given one point on one image, expect to find the other point on the other image.
    """
    # Given
    row = 49
    col_0 = 49
    col_1 = 40
    image_left, image_right = np.zeros((100,100)), np.zeros((100,100))
    image_left[row,col_0] = 1.0
    image_right[row,col_1] = 1.0
    left_point = np.vstack((col_0, row, 1), dtype=int) 
    expected_right_point = np.vstack((col_1, row, 1), dtype=int)
    # When
    right_point = dlt.ComputePointCorrespondences(image_left, image_right, left_point)
    # Then
    assert (right_point == expected_right_point).all()

# The following Stereo-Camera Pair have only relative Translation, no Rotation
P_Left = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0],
                   [0.0, 0.0, 1.0, 0.0]])

P_Right = np.array([[1.0, 0.0, 0.0, -0.5],
                    [0.0, 1.0, 0.0,  0.0],
                    [0.0, 0.0, 1.0,  0.0]])

def test_Given3DPointAtInfinityAndStereoCameras_Expect2DProjectionsAreEqual():
    """ Test that, for Stereo Cameras with only relative translation (no rotation), Points-At-Infinity appear at the same 2D Image Point on both Cameras
        3D Points at infinity have homogenous coordinate 0 (hence normalizing sends all values to infinity)
    """
    X = np.vstack((0.25, 0.0, 100, 0.0)) # 4th value 0, means "point at infinity"
    x_left  = P_Left @ X
    x_right = P_Right @ X

    assert (x_left == x_right).all() # Points at Infinity appear at the same 2D Image Point

def test_TriangulateSpacePoints_GivenEqual2DProjectionsAndStereoCameras_Expect3DPointAtInfinity():
    """ Test TriangulateSpacePoints()
        The same point in both images triangulates to a point at infinity.
    """
    x = np.vstack((0.0, 0.0, 1.0))
    X = dlt.TriangulateSpacePoints(x, P_Left, x, P_Right)

    assert (X[-1] == 0) # homogenous coordinate is zero

def test_TriangulateSpacePoints_Given2DProjectionsAndStereoCameras_ExpectOriginal3DPoint():
    """ Test TriangulateSpacePoints()
        A 3D Point projected into two Image Points triangulates back to the original 3D Point.
    """
    X_orig = np.vstack((0.25, 0.0, 100.0, 1.0))
    x_left  = P_Left @ X_orig
    x_left  = x_left/x_left[-1]    # Normalize by dividing by homogenous coordinate
    x_right = P_Right @ X_orig
    x_right  = x_right/x_right[-1] # Normalize by dividing by homogenous coordinate

    X_calc = dlt.TriangulateSpacePoints(x_left, P_Left, x_right, P_Right)
    X_calc = X_calc / X_calc[-1] # Normalize by dividing by homogenous coordinate

    assert (X_calc == pytest.approx(X_orig, abs=1e-13))




