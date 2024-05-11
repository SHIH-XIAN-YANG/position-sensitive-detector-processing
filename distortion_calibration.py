import numpy as np
from scipy.optimize import curve_fit
import math
import pandas as pd
from distortion  import Distortion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# t1 = np.array([213,904,194]).T 
# t2 = np.array([247,904,194]).T
t1 = np.array([203,893,184]).T 
t2 = np.array([257,893,178]).T
def img2world(x1,y1,x2,y2):
    """
    The function convert psd data from image frame to world frame
    Unit: (mm)
    Args:
        x and y data from PSD 1 and PSD 2 (Unit: (mm))

    Returns:
        world coordinate (X_w, Y_w, Z_w)
    """

    
    x90 = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
    y180 = np.array([[-1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    r = np.dot(y180, x90)  # Corrected matrix multiplication
    # t1 = t1.reshape(-1, 1)
    # t2 = t2.reshape(-1, 1)

    # Concatenate the translation vector to the rotation matrix
    world2camera = np.hstack((r, np.expand_dims(t1, axis=1)))
    world2camera1 = np.vstack(world2camera1, np.array([0, 0, 0, 1]))
    world2camera = np.hstack((r, np.expand_dims(t2, axis=1)))
    world2camera2 = np.vstack(world2camera2, np.array([0, 0, 0, 1]))
    print(world2camera2)
    f = 9 # 9mm
    intrinsic = np.array([[f, 0, 0],
                          [0, f, 0],
                          [0, 0, 1]])
    camera2img = np.hstack(intrinsic, np.zeros((3, 1)))
    camera2img = np.vstack((camera2img, np.array([0, 0, 0, 1])))
    
    
    z_w = compute_depth(x1,x2)
    world_coordinate = z_w * np.dot(np.linalg.inv(world2camera1), np.dot(np.linalg.inv(intrinsic), np.array([[x1],[y1],[1], [1]])))
    # position = z_w * np.dot(np.linalg.inv(r), np.dot(np.linalg.inv(intrinsic), np.array([[x1],[y1],[1]]))) - np.dot(np.linalg.inv(r), t1)
    # x_w = position[0]
    # y_w = position[1]
    # z_w = position[2]
    return world_coordinate

def world2img(x_w,y_w,z_w,t):

    # t1 = [213,904,194] 
    # t2 = [247,904,194]

    
    x90 = np.array([[1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]])
    y180 = np.array([[-1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
    r = np.dot(y180, x90)  # Corrected matrix multiplication
    f = 6 # 9mm
    intrinsic = np.array([[f, 0, 0],
                        [0, f, 0],
                        [0, 0, 1]])
    
    world2camera = np.hstack((r, np.expand_dims(t, axis=1)))
    world2camera = np.vstack((world2camera, np.array([0, 0, 0, 1])))
    
    world_coordinate = np.array([x_w, y_w, z_w, 1]).T 
    
    camera_coordinate = np.dot(world2camera, world_coordinate)
    
    camera2img = np.hstack((intrinsic, np.zeros((3, 1))))
    camera2img = np.vstack((camera2img, np.array([0, 0, 0, 1])))
    
    img_coordinate = np.dot(camera2img, camera_coordinate)
    x_img = img_coordinate[0] / img_coordinate[2]
    y_img = img_coordinate[1] / img_coordinate[2]
    z_img = 1
    return np.array([x_img, y_img, z_img])


def compute_depth(xa, xb):
    # Unit: (mm)
    # z(depth) = b*f/(xa-xb)
    b = 54
    f = 9
    return b * f / (xa - xb)


def load_psd_data(filename):
    # Load the CSV file
    df = pd.read_csv(filename)

    # Extract 'x1', 'y1', 'x2', and 'y2' columns
    x1 = df['x1(mm)'].values
    y1 = df['y1(mm)'].values
    x2 = df['x2(mm)'].values
    y2 = df['y2(mm)'].values
    x_ideal = df['x(mm)'].values
    y_ideal = df['y(mm)'].values
    z_ideal = df['z(mm)'].values

    return x1, y1, x2, y2, x_ideal, y_ideal, z_ideal

def load_HRSS_trajectory(self,path_dir:str):
        try:
            data = np.genfromtxt(path_dir, delimiter=',')
        except:
            return None
        
        # cartesian command(mm)
        x_c = data[:,0]/1000
        y_c = data[:,1]/1000
        z_c = data[:,2]/1000

        # cartesian command(degree)
        pitch_c = data[:, 3]/1000
        yaw_c = data[:, 4]/1000
        roll_c = data[:,5]/1000
        return x_c, y_c, z_c, pitch_c, yaw_c, roll_c



def main():
    psd_data_filename = "./black_extrinsic_2024_5_11_19_52.csv"
    rt605_data_filename = "./points.txt"
    x1_observed, y1_observed,x2_observed, y2_observed,x_hrss,y_hrss,z_hrss = load_psd_data(psd_data_filename)
    # x_hrss,y_hrss,z_hrss,a_hrss,b_hrss,c_hrss = load_HRSS_trajectory(rt605_data_filename)
    
    #尋找機械手臂的輸出位置資訊與PSD資料對其

    print((x1_observed.shape))

    
    
    
    #TODO: convert ideal path x,y,z from worl frame to image frame
    ideal_img_coordinate = np.zeros((x_hrss.shape[0], 3))
    for i,(x,y,z) in enumerate(zip(x_hrss,y_hrss,z_hrss)):
        ideal_img_coordinate[i] = world2img(x,y,z, t1)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(ideal_img_coordinate[:,0], ideal_img_coordinate[:,1], ideal_img_coordinate[:,2], c='r', marker='o')  # 'c' sets the color, 'marker' sets the marker style
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    plt.grid = True
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter3D(x1_observed, y1_observed, np.ones((x1_observed.shape[0], 1)), c='r', marker='o')  # 'c' sets the color, 'marker' sets the marker style
    ax.set_xlabel('X ')
    ax.set_ylabel('Y ')
    ax.set_zlabel('Z ')
    plt.grid = True
    plt.show()

    # distortion = Distortion(x1_observed, y1_observed,ideal_img_coordinate[:,2], ideal_img_coordinate[:,0],ideal_img_coordinate[:,1],ideal_img_coordinate[:,2])
    # distortion.distortion__params_identification()
    # distortion.plot_distortion(show=True)

    # distortion.write_distortion_parameters_to_file("distortion_parameters.dp")

    # data = distortion.undistort(data)
    

if __name__=="__main__":
    main()





