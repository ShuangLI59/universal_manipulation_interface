import numpy as np
import json
import os
import pandas as pd
import pdb
import glob
from plot_xyz import polt_fn
from scipy.spatial.transform import Rotation

## this convert the camera trajectory from the SLAM coordinate to the ARUCO tag coordinate
tx_slam_tag_path = '/store/real/shuang/GoPro_mapping1/demos/mapping/tx_slam_tag.json'

gopro_path = '/store/real/shuang/GoPro_mapping1/demos/'
camera_trajectory_files = glob.glob(gopro_path + '*/camera_trajectory.csv')
camera_trajectory_files.sort()

csv_paths = {'/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.14.21.001633/camera_trajectory.csv': 'metal_cup1',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.15.10.234150/camera_trajectory.csv': 'metal_cup2',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.15.41.348567/camera_trajectory.csv': 'metal_cup3',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.16.07.741600/camera_trajectory.csv': 'metal_cup4',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.16.32.766600/camera_trajectory.csv': 'metal_cup5',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.17.17.361150/camera_trajectory.csv': 'blue_cup1',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.17.47.841600/camera_trajectory.csv': 'blue_cup2',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.18.12.899967/camera_trajectory.csv': 'blue_cup3',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.18.51.471833/camera_trajectory.csv': 'blue_cup4',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.19.17.447783/camera_trajectory.csv': 'blue_cup5',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.20.12.569517/camera_trajectory.csv': 'blue_cup6',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.20.37.360950/camera_trajectory.csv': 'blue_cup7',
            '/store/real/shuang/GoPro_mapping1/demos/demo_C3444250470334_2024.12.12_07.20.59.766667/camera_trajectory.csv': 'no_cup',
            '/store/real/shuang/GoPro_mapping1/demos/mapping/camera_trajectory.csv': 'mapping'}


output_dir_csv = 'result_slam_aruco_coordinate_csv'
output_dir_xyz_plot = 'result_slam_aruco_coordinate_xyz_plot'

os.makedirs(output_dir_csv, exist_ok=True)
os.makedirs(output_dir_xyz_plot, exist_ok=True)


def main():
    # SLAM map origin to table tag transform
    tx_slam_tag = np.array(json.load(
        open(tx_slam_tag_path, 'r')
        )['tx_slam_tag']
    )
    tx_tag_slam = np.linalg.inv(tx_slam_tag)
    
    for csv_path, filetag in csv_paths.items():
        csv_df = pd.read_csv(csv_path)
        cam_pos = csv_df[['x', 'y', 'z']].to_numpy()
        cam_rot_quat_xyzw = csv_df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
        cam_rot = Rotation.from_quat(cam_rot_quat_xyzw)
        cam_pose = np.zeros((cam_pos.shape[0], 4, 4), dtype=np.float32)
        cam_pose[:,3,3] = 1
        cam_pose[:,:3,3] = cam_pos
        cam_pose[:,:3,:3] = cam_rot.as_matrix()
        tx_slam_cam = cam_pose
        tx_tag_cam = tx_tag_slam @ tx_slam_cam

        # Extract the transformed position and orientation
        new_cam_pos = tx_tag_cam[:, :3, 3]
        new_cam_rot = Rotation.from_matrix(tx_tag_cam[:, :3, :3])
        new_cam_rot_quat = new_cam_rot.as_quat()  # Output is [q_x, q_y, q_z, q_w]

        # Prepare the new DataFrame
        output_df = csv_df.copy()
        output_df['x'] = new_cam_pos[:, 0]
        output_df['y'] = new_cam_pos[:, 1]
        output_df['z'] = new_cam_pos[:, 2]
        output_df['q_x'] = new_cam_rot_quat[:, 0]
        output_df['q_y'] = new_cam_rot_quat[:, 1]
        output_df['q_z'] = new_cam_rot_quat[:, 2]
        output_df['q_w'] = new_cam_rot_quat[:, 3]

        
        # Save to a new CSV file
        output_csv_path = f'%s/%s_%s.csv' % (output_dir_csv, csv_path.split('/')[-2], filetag)
        print(f"Transformed data saved to {output_csv_path}")
        output_df.to_csv(output_csv_path, index=False)

        
        # Extracting the necessary data
        output_xyz_plot_path = f'%s/%s_%s' % (output_dir_xyz_plot, csv_path.split('/')[-2], filetag)
        timestamps = output_df['timestamp']
        x = output_df['x']
        y = output_df['y']
        z = output_df['z']
        polt_fn(timestamps, x, y, z, output_xyz_plot_path)


main()