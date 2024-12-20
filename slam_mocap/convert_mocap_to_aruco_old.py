import os
import pickle
import pdb
import numpy as np
# from slam_mocap.plot_xyz import plot_fn
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

def plot_fn(timestamps, x, y, z, filetag, y_min=-0.5, y_max=0.6):
    # Determine the global y-axis limits
    
    # Plotting X Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, x, label='X Coordinate', alpha=0.8)
    plt.title('X Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('X')
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{filetag}_X_Coordinate_over_Time.png')

    # Plotting Y Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, y, label='Y Coordinate', alpha=0.8, color='orange')
    plt.title('Y Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Y')
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{filetag}_Y_Coordinate_over_Time.png')

    # Plotting Z Coordinate
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, z, label='Z Coordinate', alpha=0.8, color='green')
    plt.title('Z Coordinate over Time')
    plt.xlabel('Timestamp')
    plt.ylabel('Z')
    plt.ylim(y_min, y_max)  # Set consistent y-axis limits
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{filetag}_Z_Coordinate_over_Time.png')
    
    
# Specify the directory containing the .pkl files
mocap_data_path = "mocap_data"


def quaternion_to_rotation_matrix(w, x, y, z):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.
    """
    # Normalize the quaternion
    norm = np.sqrt(w**2 + x**2 + y**2 + z**2)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    # Rotation matrix computation
    rot_matrix = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),       1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),     1 - 2 * (x**2 + y**2)]
    ])
    return rot_matrix

def get_transformation_matrix_from_array(array):
    """
    Get a 4x4 transformation matrix from an array of 7 elements:
    [x, y, z, w, qx, qy, qz].
    """
    assert len(array) == 7, "Input array must have exactly 7 elements: [x, y, z, w, qx, qy, qz]"
    
    # Extract position and quaternion
    x, y, z, w, qx, qy, qz = array

    # Create the rotation matrix
    rotation_matrix = quaternion_to_rotation_matrix(w, qx, qy, qz)

    # Create the 4x4 transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]

    return transformation_matrix



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
            'umi_gripper_2024-12-11_23-20-39_no_cup_GX010055_no_cup.pkl']

    # Iterate over each file in the directory
    plan = []
    for filename in files:
        filepath = os.path.join(mocap_data_path, filename)
        with open(filepath, 'rb') as file:
            print(f"Loading {filepath}")
            data = pickle.load(file)
            
            # pdb.set_trace()
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
            aruco_data = np.array([umi_aruco_x, umi_aruco_y, umi_aruco_z, umi_aruco_qw, umi_aruco_qx, umi_aruco_qy, umi_aruco_qz])
            aruco_position = aruco_data[:3]  # Aruco tag XYZ position
            aruco_quaternion = aruco_data[3:]  # Quaternion (x, y, z, w)
            aruco_rot = R.from_quat(aruco_quaternion, scalar_first=True)
            
            
            posrot = [tem[0] for tem in data['umi_gripper']]
            posrot = np.stack(posrot, axis=0)
            
            relative_posrot = []
            
            umi_poses_mocap = []
            umi_poses_artag = []
            for posrot_each in posrot:
                umi_position = posrot_each[:3]
                umi_quaternion = posrot_each[3:]
                umi_rot = R.from_quat(umi_quaternion, scalar_first=True)
                
                ############################################################################################################
                ## v1
                # axis_angle = umi_rot.as_rotvec() # convert quaternion to axis-angle
                # relative_posrot.append(np.hstack((umi_position, axis_angle)))
                
                ############################################################################################################
                ## v2
                # Compute relative position
                relative_position = aruco_rot.inv().apply(umi_position - aruco_position)
                # pdb.set_trace()

                # Compute relative rotation
                relative_rotation = aruco_rot.inv() * umi_rot
                
                # pdb.set_trace()
                #####################################
                R_opt = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).transpose()
                t_opt = np.array([0.0, 0.0, 0.0])
                
                relative_position = R_opt @ relative_position
                relative_rotation = R_opt.transpose() @ relative_rotation.as_matrix()
                relative_rotation = R.from_matrix(relative_rotation)
                #####################################
                
                # pdb.set_trace()
                # Convert relative rotation to axis-angle
                axis_angle = relative_rotation.as_rotvec()
                relative_posrot.append(np.hstack((relative_position, axis_angle)))
                
                # relative_quaternion = relative_rotation.as_quat()
                # relative_posrot.append(np.hstack((relative_position, relative_quaternion)))
            
                # umi_poses_mocap.append(get_transformation_matrix_from_array(posrot_each))
                # umi_poses_artag.append(get_transformation_matrix_from_array(np.hstack((relative_position, relative_quaternion))))
                ############################################################################################################
            
            gripper_with_axis_angle = np.stack(relative_posrot)
            plan.append({'episode_timestamps': timestamps, 'grippers': [ {'tcp_pose': gripper_with_axis_angle} ]})
            # pdb.set_trace()
            
            
            # umi_poses_mocap = np.stack(umi_poses_mocap)
            # with open("umi_poses_mocap.pkl", "wb") as pickle_file:
            #     pickle.dump(umi_poses_mocap, pickle_file)
            
            # umi_poses_artag = np.stack(umi_poses_artag)
            # with open("umi_poses_artag.pkl", "wb") as pickle_file:
            #     pickle.dump(umi_poses_artag, pickle_file)
            # pdb.set_trace()
            
            # ######################################################################################################
            # # Fix the rotation and translation
            # ######################################################################################################
            # R_opt = np.array([[0, 0, -1],
            #       [1, 0, 0],
            #       [0, -1, 0]])
            # t_opt = np.array([0.0, 0.0, 0.0])
            
            # positions = gripper_with_axis_angle[:, :3]
            # rotations = gripper_with_axis_angle[:, 3:]
            
            # # Transform positions in System 1 to System 2
            # transformed_positions = (R_opt @ positions.T).T + t_opt

            # # Compute magnitudes and directions of axis-angle rotations
            # magnitudes = np.linalg.norm(rotations, axis=1, keepdims=True)
            # directions = rotations / (magnitudes + 1e-8)  # Normalize to avoid division by zero

            # # Apply the optimal rotation R_opt to the directions
            # transformed_directions = (R_opt @ directions.T).T
            # transformed_rotations = transformed_directions * magnitudes

            # # Combine transformed positions and rotations
            # transformed_data = np.hstack((transformed_positions, transformed_rotations))

            # ######################################################################################################
            
            
            
            # plan.append({'episode_timestamps': timestamps, 'grippers': [ {'tcp_pose': transformed_data} ]})
            
    return plan
            


def main():
    files = os.listdir(mocap_data_path)
    files.sort()

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
            'umi_gripper_2024-12-11_23-20-39_no_cup_GX010055_no_cup.pkl',
            'umi_gripper_umi_aruco_2024-12-11_23-05-22_mapping1_GX010022_mapping1.pkl']


    output_dir_xyz_plot = 'result_mocap_aruco_coordinate_xyz_plot_1219'
    os.makedirs(output_dir_xyz_plot, exist_ok=True)

    # Iterate over each file in the directory
    for filename in files:
        filepath = os.path.join(mocap_data_path, filename)
        with open(filepath, 'rb') as file:
            print(f"Loading {filepath}")
            data = pickle.load(file)
            
            init_time = data['umi_gripper'][0][1]
            
            posrot = [tem[0] for tem in data['umi_gripper']]
            posrot = np.stack(posrot, axis=0)
            timestamps = [tem[1]-init_time for tem in data['umi_gripper']]
            
            output_xyz_plot_path = f'%s/%s' % (output_dir_xyz_plot, filepath.split('/')[-1][:-4])
            # plot_fn(timestamps, posrot[:,0]-umi_aruco_x, posrot[:,1]-umi_aruco_y, posrot[:,2]-umi_aruco_z, output_xyz_plot_path)
            
            # plot_fn(timestamps, posrot[:,2]-umi_aruco_z, posrot[:,0]-umi_aruco_x, posrot[:,1]-umi_aruco_y, output_xyz_plot_path)
            plot_fn(timestamps, posrot[:,0], posrot[:,1], posrot[:,2], output_xyz_plot_path)
            
            
# main()