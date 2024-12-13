import os
import pickle
import pdb
import numpy as np
from plot_xyz import plot_fn

# Specify the directory containing the .pkl files
mocap_data_path = "mocap_data"

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


mapping_filepath = os.path.join(mocap_data_path, 'umi_gripper_umi_aruco_2024-12-11_23-05-22_mapping1_GX010022_mapping1.pkl')
with open(mapping_filepath, 'rb') as file:
    print(f"Loading {mapping_filepath}")
    data = pickle.load(file)
    umi_aruco_position = [tem[0] for tem in data['umi_aruco']]
    umi_aruco_position = np.stack(umi_aruco_position)
    umi_aruco_x = np.mean(umi_aruco_position[:,0])
    umi_aruco_y = np.mean(umi_aruco_position[:,1])
    umi_aruco_z = np.mean(umi_aruco_position[:,2])
    print(f"umi_aruco_x: {umi_aruco_x}, umi_aruco_y: {umi_aruco_y}, umi_aruco_z: {umi_aruco_z}")
    

output_dir_xyz_plot = 'result_mocap_aruco_coordinate_xyz_plot'
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
        
        plot_fn(timestamps, posrot[:,2]-umi_aruco_z, posrot[:,0]-umi_aruco_x, posrot[:,1]-umi_aruco_y, output_xyz_plot_path)
        
        
        