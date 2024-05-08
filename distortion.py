import numpy as np
from scipy.optimize import curve_fit
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def distortion_func(r,x,y,k1,k2,k3,p1,p2):
    r = math.sqrt(x**2 + y**2)
    x_distortion = x * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + 2 * p1 * x * y + p2 * (r**2 + 2 * x**2)
    y_distortion = y * (1 + k1 * r**2 + k2 * r**4 + k3 * r**6) + p1 * (r**2 + 2 * y**2) + 2 * p2 * x * y
    return (x_distortion, y_distortion)

def objective_function(parames,ideal):
    x, y = ideal
    
    k1,k2,k3,p1,p2 = parames
    x_predicted, y_predicted = distortion_func(x, y, k1, k2, k3, p1, p2)

    # error_x = x_predicted - x_observed
    # error_y = y_predicted - y_observed
    return (x_predicted, y_predicted)

class Distortion():
    k1: float
    k2: float 
    k3: float
    p1: float
    p2: float

    params_initial_guess: list

    x_observed: np.ndarray
    y_observed: np.ndarray
    z_observed: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


    def __init__(self,x_observed,y_observed,z_observed, x,y,z):
        self.params_initial_guess = (0.01, 0.01, 0.01, 0.0001, 0.0001) # [k1, k2, k3, p1, p2]
        self.objective_funct = objective_function
        self.distortion_func = distortion_func

        self.x_observed = x_observed
        self.y_observed = y_observed
        self.z_observed = z_observed

        self.x = x
        self.y = y
        self.z = z

    def distortion__params_identification(self):
        r = [math.sqrt(a**2 + b**2) for a,b in zip(self.x,self.y)]

        params_opt, covariance = curve_fit(objective_function, np.concatenate((self.x, self.y, r)), np.concatenate((self.x_observed, self.y_observed)), p0=self.params_initial_guess)

        self.k1, self.k2, self.k3, self.p1, self.p2 = params_opt

    

    def undistort(self, distorted_data):
        """
        Undistort points using radial and tangential distortion parameters.
        
        Args:
            distorted_points (numpy array): Array of distorted points of shape (N, 2).
            k1, k2, k3 (float): Radial distortion coefficients.
            p1, p2 (float): Tangential distortion coefficients.
        
        Returns:
            undistorted_points (numpy array): Array of undistorted points of shape (N, 2).
        """
        undistorted_points = np.zeros_like(distorted_data)
        
        for i, (x, y) in enumerate(distorted_data):
            # Normalize pixel coordinates
            x_norm = (x - self.p1) / (1 + self.k1*x + self.k2*y)
            y_norm = (y - self.p2) / (1 + self.k1*x + self.k2*y)
            
            # Iterative distortion correction
            for _ in range(5):  # Iterative correction for better accuracy
                r2 = x_norm**2 + y_norm**2
                distortion_radial = 1 + self.k1*r2 + self.k2*r2**2 + self.k3*r2**3
                delta_x = 2*self.p1*x_norm*y_norm + self.p2*(r2 + 2*x_norm**2)
                delta_y = self.p1*(r2 + 2*y_norm**2) + 2*self.p2*x_norm*y_norm
                x_norm = (x - delta_x) / distortion_radial
                y_norm = (y - delta_y) / distortion_radial
            
            # Denormalize pixel coordinates
            undistorted_points[i] = [x_norm, y_norm]
        
        return undistorted_points
    
    def plot_distortion(self, show=False):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        x_corrected, y_corrected = self.undistort(self.x_observed, self.y_observed)

        ax.scatter(self.x_observed, self.y_observed, self.z_observed, color='b', label='observed')
        ax.scatter(self.x, self.y, self.z, c='r',label='real')
        ax.scatter(x_corrected, y_corrected, self.z_observed,c='g', label='corrected') 

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        plt.legend()
        plt.show()
        return fig