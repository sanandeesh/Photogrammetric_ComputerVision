"""
    OXTS Utilities
    Supplements VIT-EKF by providing utilities for parsing KITTI OXTS Odometry Data.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.lib.recfunctions as rf
from types import SimpleNamespace
from pyproj import Transformer
from datetime import datetime
from pathlib import Path

"""
lat:   latitude of the oxts-unit (deg)
lon:   longitude of the oxts-unit (deg)
alt:   altitude of the oxts-unit (m)
roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
vn:    velocity towards north (m/s)
ve:    velocity towards east (m/s)
vf:    forward velocity, i.e. parallel to earth-surface (m/s)
vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
ay:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
af:    forward acceleration (m/s^2)
al:    leftward acceleration (m/s^2)
au:    upward acceleration (m/s^2)
wx:    angular rate around x (rad/s)
wy:    angular rate around y (rad/s)
wz:    angular rate around z (rad/s)
wf:    angular rate around forward axis (rad/s)
wl:    angular rate around leftward axis (rad/s)
wu:    angular rate around upward axis (rad/s)
pos_accuracy:  velocity accuracy (north/east in m)
vel_accuracy:  velocity accuracy (north/east in m/s)
navstat:       navigation status (see navstat_to_string)
numsats:       number of satellites tracked by primary GPS receiver
posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
"""
oxts_dtype = np.dtype([
    ('lat', 'f8'), 
    ('lon', 'f8'), 
    ('alt', 'f8'),
    ('roll', 'f8'), 
    ('pitch', 'f8'), 
    ('yaw', 'f8'),
    ('vn', 'f8'), 
    ('ve', 'f8'), 
    ('vf', 'f8'),
    ('vl', 'f8'), 
    ('vu', 'f8'), 
    ('ax', 'f8'),
    ('ay', 'f8'), 
    ('az', 'f8'), 
    ('af', 'f8'),
    ('al', 'f8'), 
    ('au', 'f8'), 
    ('wx', 'f8'),
    ('wy', 'f8'), 
    ('wz', 'f8'), 
    ('wf', 'f8'),
    ('wl', 'f8'), 
    ('wu', 'f8'), 
    ('pos_accuracy', 'f8'),
    ('vel_accuracy', 'f8'), 
    ('navstat', 'f8'), 
    ('numsats', 'i4'),
    ('posmode', 'i4'), 
    ('velmode', 'i4'), 
    ('orimode', 'i4')  
])

# Get OXTS TimeStamps
def GetOXTSTimeStamps(data_path):
    """Get OXTS TimeStamps
       
    Args:
        data_path: path to root directory of the recording.

    Returns:
        TimeStamps np.array
    """
    data_path = data_path + 'oxts/timestamps.txt' 
    time_stamps_list = []
    first_time = None
    with open(data_path, "r") as file:
        for line in file:
            line_str = line.strip()
            if not line_str:
                continue
            line_str = line_str[:26]
            # Parse the string (automatically truncates the last 3 digits to fit microseconds)
            current_time = datetime.strptime(line_str, "%Y-%m-%d %H:%M:%S.%f")
            # Capture the first timestamp as your zero baseline
            if first_time is None:
                first_time = current_time
            # Subtracting datetimes creates a timedelta object
            elapsed_delta = current_time - first_time
            # Convert the full difference into total seconds with float precision
            elapsed_seconds = elapsed_delta.total_seconds()
            time_stamps_list.append(elapsed_seconds)
            # Print with 6 decimal places
            print(f"{elapsed_seconds:.6f}", elapsed_seconds)
    
    return np.array(time_stamps_list)

# Get OXTS Data Sample
def GetOXTSSample(data_path, index):
    """Get OXTS Sample
       
    Args:
        data_path: path to root directory of the recording.
        index: recording sample index
    Returns:
        OXTS Sample
    """
    width = 10
    padded_number = str(index).zfill(width)
    print('OXTS Sample ', index)
    oxts_data_name  = data_path + '/oxts/data/' + padded_number + '.txt'
    # 
    data = np.loadtxt(oxts_data_name)
    structured_array = rf.unstructured_to_structured(data, oxts_dtype)

    return structured_array

def AddOXTSData(data_path, oxts_data):
    """Get OXTS Data
       
    Args:
        data_path: path to root directory of the recording.
        oxts_data: struct to update with data
    Returns:
        oxts_data updated
    """    
    dir_path = Path(data_path + 'oxts/data/')
    num_samples = sum(1 for item in dir_path.iterdir() if item.is_file())
    #
    num_samples_ts = len(oxts_data.local_timestamps)
    assert num_samples == num_samples_ts
    oxts_data.num_samples = num_samples
    #
    lat = []
    lon = []
    yaw = []
    vn = []
    ve = []
    vf = []
    vl = []
    wl = []
    wu = []
    for index in range(num_samples):
        oxts_sample = GetOXTSSample(data_path, index)
        lat.append(oxts_sample['lat'])
        lon.append(oxts_sample['lon'])
        yaw.append(oxts_sample['yaw'])
        vn.append(oxts_sample['vn'])
        ve.append(oxts_sample['ve'])
        vf.append(oxts_sample['vf'])
        vl.append(oxts_sample['vl'])
        wl.append(oxts_sample['wl'])
        wu.append(oxts_sample['wu'])
    
    oxts_data.lat = np.asarray(lat)
    oxts_data.lon = np.asarray(lon)
    oxts_data.yaw = np.asarray(yaw)
    oxts_data.vn = np.asarray(vn)
    oxts_data.ve = np.asarray(ve)
    oxts_data.vf = np.asarray(vf)
    oxts_data.vl = np.asarray(vl)
    oxts_data.wl = np.asarray(wl)
    oxts_data.wu = np.asarray(wu)

    return oxts_data

def AddNorthingEasting(oxts_data):
    """Compute Local Northing\Easting from lat\lon data.

    Args:
        oxts_data:
    Returns:
        oxts_data updated
    """     
    # EPSG:4326 
    # EPSG:25832  
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    easting, northing = transformer.transform(oxts_data.lon, oxts_data.lat)
    oxts_data.local_easting = np.array(easting) - easting[0]
    oxts_data.local_northing = np.array(northing) - northing[0]

    return oxts_data

def GetOXTSDataFull(data_path):
    """Get OXTS Data Full
    Root function for extraction OXTS data
       
    Args:
        data_path: path to root directory of the recording.
    Returns:
        oxts_data full
    """ 
    oxts_data = SimpleNamespace()
    #
    oxts_data.local_timestamps = GetOXTSTimeStamps(data_path)
    oxts_data = AddOXTSData(data_path, oxts_data)

    return oxts_data

if __name__ == "__main__":
    data_path = '2011_09_26/2011_09_26_drive_0001_sync/'
    oxts_data = GetOXTSDataFull(data_path)
    # 
    plt.figure(figsize=(30, 18))
    plt.subplot(2, 1, 1)
    gt_pos_sctr = plt.scatter(oxts_data.local_easting, 
                              oxts_data.local_northing, color='blue', label='GroundTruth')
    plt.grid(True)
    plt.axis('equal')
    # 
    rad2deg = 180.0/np.pi
    x_values = np.arange(oxts_data.num_samples)
    plt.subplot(2, 1, 2)
    pred_yaw_plt = plt.plot(x_values, rad2deg*oxts_data.yaw, color='blue') 
    plt.grid(True)
    plt.title('GroundTruth Yaw')
    # 
    plt.show()