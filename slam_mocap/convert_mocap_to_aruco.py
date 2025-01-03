import os
import pickle
import pdb
import numpy as np
# from slam_mocap.plot_xyz import plot_fn
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import pandas as pd

from slam_mocap.plot_xyz import plot_fn
# Specify the directory containing the .pkl files
mocap_data_path = "mocap_data"



mapping_filepath = os.path.join(mocap_data_path, 'umi_gripper_umi_aruco_2024-12-11_23-05-22_mapping1_GX010022_mapping1.pkl')
with open(mapping_filepath, 'rb') as file:
    print(f"Loading {mapping_filepath}")
    data = pickle.load(file)
    umi_aruco_position = [tem[0] for tem in data['umi_aruco']]
    umi_aruco_position = np.stack(umi_aruco_position)
    
    umi_aruco_x = np.mean(umi_aruco_position[:,0])
    umi_aruco_y = np.mean(umi_aruco_position[:,1])
    umi_aruco_z = np.mean(umi_aruco_position[:,2])
    umi_aruco_qw = np.mean(umi_aruco_position[:,3])
    umi_aruco_qx = np.mean(umi_aruco_position[:,4])
    umi_aruco_qy = np.mean(umi_aruco_position[:,5])
    umi_aruco_qz = np.mean(umi_aruco_position[:,6])
    
    print(f"umi_aruco_x: {umi_aruco_x}, umi_aruco_y: {umi_aruco_y}, umi_aruco_z: {umi_aruco_z}")
    print(f"umi_aruco_qw: {umi_aruco_qw}, umi_aruco_qx: {umi_aruco_qx}, umi_aruco_qy: {umi_aruco_qy}, umi_aruco_qz: {umi_aruco_qz}")
    

def get_mocap_plan():
    files = ['umi_gripper_2024-12-11_23-13-56_metal_cup1_GX010041_metal_cup1.pkl',
            'umi_gripper_2024-12-11_23-14-49_metal_cup2_GX010042_metal_cup2.pkl',
            'umi_gripper_2024-12-11_23-15-13_metal_cup3_GX010043_metal_cup3.pkl',
            'umi_gripper_2024-12-11_23-15-37_metal_cup4_GX010044_metal_cup4.pkl',
            'umi_gripper_2024-12-11_23-16-02_metal_cup5_GX010045_metal_cup5.pkl',
            'umi_gripper_2024-12-11_23-16-50_blue_cup1_GX010046_blue_cup1.pkl',
            'umi_gripper_2024-12-11_23-17-18_blue_cup2_GX010048_blue_cup2.pkl',
            'umi_gripper_2024-12-11_23-17-47_blue_cup3_GX010049_blue_cup3.pkl',
            'umi_gripper_2024-12-11_23-18-23_blue_cup4_GX010050_blue_cup4.pkl',
            'umi_gripper_2024-12-11_23-19-42_blue_cup6_GX010052_blue_cup6.pkl',
            'umi_gripper_2024-12-11_23-20-06_blue_cup7_GX010054_blue_cup7.pkl',
            'umi_gripper_2024-12-11_23-20-39_no_cup_GX010055_no_cup.pkl'
            ]

    # Iterate over each file in the directory
    plan = []
    for filename in files:
        filepath = os.path.join(mocap_data_path, filename)
        with open(filepath, 'rb') as file:
            print(f"Loading {filepath}")
            data = pickle.load(file)
            
            ######################################################################################################
            ## get time stamps
            ######################################################################################################
            init_time = data['umi_gripper'][0][1]
            # timestamps = [tem[1]-init_time for tem in data['umi_gripper']]
            timestamps = [tem[1] for tem in data['umi_gripper']]
            timestamps = np.stack(timestamps)
            
            ######################################################################################################
            # Convert UMI to Aruco coordinate
            ######################################################################################################
            
            ## aruco position and rotation
            aruco_posrot = np.array([umi_aruco_x, umi_aruco_y, umi_aruco_z, umi_aruco_qw, umi_aruco_qx, umi_aruco_qy, umi_aruco_qz])
            aruco_position = aruco_posrot[:3]  # Aruco tag XYZ position
            aruco_quaternion_wxyz = aruco_posrot[3:]  # Quaternion (w, x, y, z)
            aruco_rot = R.from_quat(aruco_quaternion_wxyz, scalar_first=True)
            
            ## umi position and rotation
            umi_posrot = [tem[0] for tem in data['umi_gripper']]
            umi_posrot = np.stack(umi_posrot, axis=0)
            
            ## plot raw data
            # plot_fn(range(len(umi_posrot)), umi_posrot[:,4], filetag='ori_mocap_QX_Coordinate', y_min=-0.4, y_max=-0.25, color='blue')
            # plot_fn(range(len(umi_posrot)), umi_posrot[:,5], filetag='ori_mocap_QY_Coordinate', y_min=0.5, y_max=0.7, color='orange')
            # plot_fn(range(len(umi_posrot)), umi_posrot[:,6], filetag='ori_mocap_QZ_Coordinate', y_min=-0.7, y_max=-0.6, color='green')
            
            
            
            relative_posrot = []
            
            for umi_posrot_each in umi_posrot:
                umi_position = umi_posrot_each[:3]
                umi_quaternion_wxyz = umi_posrot_each[3:]
                umi_rot = R.from_quat(umi_quaternion_wxyz, scalar_first=True)
                # umi_rot.as_euler("xyz")
                
                #####################################
                ## umi to aruco
                # Compute relative position
                relative_position = aruco_rot.inv().apply(umi_position - aruco_position)

                # Compute relative rotation
                relative_rotation = aruco_rot.inv() * umi_rot
                
                #####################################
                ## slam aruco to mocap arocu
                R_opt = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
                
                relative_position = R_opt @ relative_position

                ## slam umi to mocap umi
                UMI_coordinates_rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
                
                relative_rotation = R_opt @ relative_rotation.as_matrix() @ UMI_coordinates_rotation
                relative_rotation = R.from_matrix(relative_rotation)
                #####################################
                
                # Convert relative rotation to axis-angle
                axis_angle = relative_rotation.as_rotvec()
                relative_posrot.append(np.hstack((relative_position, axis_angle)))
                ############################################################################################################
            
            gripper_with_axis_angle = np.stack(relative_posrot)
            plan.append({'episode_timestamps': timestamps, 'grippers': [ {'tcp_pose': gripper_with_axis_angle} ]})
            
    return plan
