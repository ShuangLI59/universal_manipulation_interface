# %%
import sys
import os
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# %%
import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
import pdb
from collections import defaultdict
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
    get_mirror_crop_slices
)
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl

from slam_mocap.convert_mocap_to_aruco import get_mocap_plan
from slam_mocap.plot_xyz import plot_fn, plot_comparison

register_codecs()

def align_sequences(reference, target, max_shift=100):
    """
    Aligns the target sequence to the reference sequence by trying different shifts.
    Computes the distance of the overlapped part for each shift.

    Parameters:
        reference (np.ndarray): The reference sequence.
        target (np.ndarray): The target sequence to be aligned.
        max_shift (int): The maximum number of shifts to try.

    Returns:
        int: The shift that results in the minimum mean squared difference.
        float: The minimum mean squared difference of the overlapped part.
    """
    best_shift = 0
    min_mse = float('inf')

    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            overlap_reference = reference[shift:]
            overlap_target = target[:len(overlap_reference)]
        else:
            overlap_reference = reference[:len(target) + shift]
            overlap_target = target[-shift:]

        overlap_length = min(len(overlap_reference), len(overlap_target))
        if overlap_length > 0:
            overlap_reference = overlap_reference[:overlap_length]
            overlap_target = overlap_target[:overlap_length]
            mse = np.mean((overlap_reference - overlap_target) ** 2)
            if mse < min_mse:
                min_mse = mse
                best_shift = shift

    return best_shift, min_mse

# %%
@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    if os.path.isfile(output):
        if click.confirm(f'Output file {output} exists! Overwrite?', abort=True):
            pass
        
    out_res = tuple(int(x) for x in out_res.split(','))

    

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
        
    out_replay_buffer = ReplayBuffer.create_empty_zarr(
        storage=zarr.MemoryStore())
    
    # dump lowdim data to replay buffer
    # generate argumnet for videos
    n_grippers = None
    n_cameras = None
    buffer_start = 0
    all_videos = set()
    vid_args = list()
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue
        
        plan = pickle.load(plan_path.open('rb'))
        rpy = R.from_rotvec(plan[0]['grippers'][0]['tcp_pose'][0][3:])
        rpy.as_euler("xyz")
        ################################################################################################################################
        mocap_plan = get_mocap_plan()
        del plan[9]
        del plan[-1]
        del mocap_plan[-1]
        assert len(plan) == len(mocap_plan)    
        ################################################################################################################################
        
        videos_dict = defaultdict(list)
        for idx, plan_episode in enumerate(plan):
            grippers = plan_episode['grippers']
            
            
            #####################################################
            grippers_mocap = mocap_plan[idx]['grippers']
            #####################################################
            # check data
            print(grippers[0]['tcp_pose'])
            print(grippers_mocap[0]['tcp_pose'])
            
            
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,0], filetag=f'0_1219_{idx}_mocap_X_Coordinate', y_min=-0.5, y_max=0.5, color='blue')
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,1], filetag=f'0_1219_{idx}_mocap_Y_Coordinate', y_min=-0.4, y_max=-0.1, color='orange')
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,2], filetag=f'0_1219_{idx}_mocap_Z_Coordinate', y_min=0, y_max=0.25, color='green')
            
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,3], filetag=f'0_1219_{idx}_mocap_RX_Coordinate', y_min=-2.5, y_max=-2, color='blue')
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,4], filetag=f'0_1219_{idx}_mocap_RY_Coordinate', y_min=-0.5, y_max=0.5, color='orange')
            # plot_fn(range(len(grippers_mocap[0]['tcp_pose'])), grippers_mocap[0]['tcp_pose'][:,5], filetag=f'0_1219_{idx}_mocap_RZ_Coordinate', y_min=-0.5, y_max=0.5, color='green')
            
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,0], filetag=f'0_1219_{idx}_slam_X_Coordinate', y_min=-0.5, y_max=0.5, color='blue')
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,1], filetag=f'0_1219_{idx}_slam_Y_Coordinate', y_min=-0.4, y_max=-0.1, color='orange')
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,2], filetag=f'0_1219_{idx}_slam_Z_Coordinate', y_min=0, y_max=0.25, color='green')
            
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,3], filetag=f'0_1219_{idx}_slam_RX_Coordinate', y_min=-2.5, y_max=-2, color='blue')
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,4], filetag=f'0_1219_{idx}_slam_RY_Coordinate', y_min=-0.5, y_max=0.5, color='orange')
            # plot_fn(range(len(grippers[0]['tcp_pose'])), grippers[0]['tcp_pose'][:,5], filetag=f'0_1219_{idx}_slam_RZ_Coordinate', y_min=-0.5, y_max=0.5, color='green')
            
            ##########################################################################################################
            ## align sequences
            ##########################################################################################################            
            # grippers_mocap[0]['tcp_pose'] = grippers_mocap[0]['tcp_pose'][-grippers[0]['tcp_pose'].shape[0]:]
            # grippers_mocap[0]['tcp_pose'] = grippers_mocap[0]['tcp_pose'][:grippers[0]['tcp_pose'].shape[0]]
            
            # Align the sequences using X Coordinate
            shift, mse = align_sequences(grippers_mocap[0]['tcp_pose'][:, 0], grippers[0]['tcp_pose'][:, 0])
            # pdb.set_trace()
            # Adjust the timestamps for alignment
            if shift > 0:
                aligned_mocap = grippers_mocap[0]['tcp_pose'][shift:]
                aligned_grippers = grippers[0]['tcp_pose'][:len(aligned_mocap)]
                aligned_gripper_width = grippers[0]['gripper_width'][:len(aligned_mocap)]
                
                assert  np.sum(plan_episode['grippers'][0]['tcp_pose'][-1]-plan_episode['grippers'][0]['demo_end_pose']) < 1e-6
                assert  np.sum(plan_episode['grippers'][0]['tcp_pose'][0]-plan_episode['grippers'][0]['demo_start_pose']) < 1e-6
                
                
                grippers[0]['demo_start_pose'] = aligned_grippers[0]
                grippers[0]['demo_end_pose'] = aligned_grippers[-1]
                
                video_start = plan_episode['cameras'][0]['video_start_end'][0]
                video_end = plan_episode['cameras'][0]['video_start_end'][0] + len(aligned_grippers)
                plan_episode['cameras'][0]['video_start_end'] = (video_start, video_end)
                
            else:
                aligned_mocap = grippers_mocap[0]['tcp_pose'][:len(grippers[0]['tcp_pose']) + shift]
                aligned_grippers = grippers[0]['tcp_pose'][-shift:]
                aligned_gripper_width = grippers[0]['gripper_width'][-shift:]
                
                assert  np.sum(plan_episode['grippers'][0]['tcp_pose'][-1]-plan_episode['grippers'][0]['demo_end_pose']) < 1e-6
                assert  np.sum(plan_episode['grippers'][0]['tcp_pose'][0]-plan_episode['grippers'][0]['demo_start_pose']) < 1e-6
                
                grippers[0]['demo_start_pose'] = aligned_grippers[0]
                grippers[0]['demo_end_pose'] = aligned_grippers[-1]
                
                video_start = plan_episode['cameras'][0]['video_start_end'][0] - shift
                video_end = plan_episode['cameras'][0]['video_start_end'][0] - shift + len(aligned_grippers)
                plan_episode['cameras'][0]['video_start_end'] = (video_start, video_end)
                
            
            if len(aligned_mocap) > len(aligned_grippers):
                # aligned_mocap = aligned_mocap[:len(aligned_grippers)]
                print(f"Warning: {len(aligned_mocap) - len(aligned_grippers)} frames of mocap data are discarded!")
                continue
                
            
            
            grippers_mocap[0]['tcp_pose'] = aligned_mocap
            grippers[0]['tcp_pose'] = aligned_grippers
            grippers[0]['gripper_width'] = aligned_gripper_width
            ##########################################################################################################            
            
            
            
            # Plot comparisons for each coordinate
            num_points = range(len(grippers_mocap[0]['tcp_pose']))
            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 0], 
                grippers[0]['tcp_pose'][:, 0], 
                filetag=f'X_Coordinate_Comparison_{idx}', 
                y_min=-0.5, 
                y_max=0.5
            )

            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 1], 
                grippers[0]['tcp_pose'][:, 1], 
                filetag=f'Y_Coordinate_Comparison_{idx}', 
                y_min=-0.4, 
                y_max=0
            )

            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 2], 
                grippers[0]['tcp_pose'][:, 2], 
                filetag=f'Z_Coordinate_Comparison_{idx}', 
                y_min=0, 
                y_max=1
            )

            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 3], 
                grippers[0]['tcp_pose'][:, 3], 
                filetag=f'RX_Coordinate_Comparison_{idx}', 
                y_min=-2.5, 
                y_max=0
            )

            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 4], 
                grippers[0]['tcp_pose'][:, 4], 
                filetag=f'RY_Coordinate_Comparison_{idx}', 
                y_min=-0.5, 
                y_max=0.5
            )

            plot_comparison(
                num_points, 
                grippers_mocap[0]['tcp_pose'][:, 5], 
                grippers[0]['tcp_pose'][:, 5], 
                filetag=f'RZ_Coordinate_Comparison_{idx}', 
                y_min=-0.5, 
                y_max=0.5
            )

            
            #####################################################
            # check that all episodes have the same number of grippers 
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers)
                
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)
                
            episode_data = dict()
            for gripper_id, gripper in enumerate(grippers):    
                eef_pose = gripper['tcp_pose']
                eef_pos = eef_pose[...,:3]
                eef_rot = eef_pose[...,3:]
                gripper_widths = gripper['gripper_width']
                demo_start_pose = np.empty_like(eef_pose)
                demo_start_pose[:] = gripper['demo_start_pose']
                demo_end_pose = np.empty_like(eef_pose)
                demo_end_pose[:] = gripper['demo_end_pose']
                
                robot_name = f'robot{gripper_id}'
                episode_data[robot_name + '_eef_pos'] = eef_pos.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle'] = eef_rot.astype(np.float32)
                episode_data[robot_name + '_gripper_width'] = np.expand_dims(gripper_widths, axis=-1).astype(np.float32)
                episode_data[robot_name + '_demo_start_pose'] = demo_start_pose
                episode_data[robot_name + '_demo_end_pose'] = demo_end_pose
                
                #####################################################
                eef_pose_mocap = grippers_mocap[gripper_id]['tcp_pose']
                # eef_pose_mocap = eef_pose_mocap[:eef_pose.shape[0]] ###################### double check this part v1
                # eef_pose_mocap = eef_pose_mocap[-eef_pose.shape[0]:] ###################### double check this part v2
                
                
                eef_pos_mocap = eef_pose_mocap[...,:3]
                eef_rot_mocap = eef_pose_mocap[...,3:]
                
                episode_data[robot_name + '_eef_pos_mocap'] = eef_pos_mocap.astype(np.float32)
                episode_data[robot_name + '_eef_rot_axis_angle_mocap'] = eef_rot_mocap.astype(np.float32)
                #####################################################
            
            
            out_replay_buffer.add_episode(data=episode_data, compressors=None)
            
            # aggregate video gen aguments
            n_frames = None
            for cam_id, camera in enumerate(cameras):
                video_path_rel = camera['video_path']
                video_path = demos_path.joinpath(video_path_rel).absolute()
                assert video_path.is_file()
                
                video_start, video_end = camera['video_start_end']
                if n_frames is None:
                    n_frames = video_end - video_start
                else:
                    assert n_frames == (video_end - video_start)
                
                videos_dict[str(video_path)].append({
                    'camera_idx': cam_id,
                    'frame_start': video_start,
                    'frame_end': video_end,
                    'buffer_start': buffer_start
                })
            buffer_start += n_frames
        
        vid_args.extend(videos_dict.items())
        all_videos.update(videos_dict.keys())
    
    print(f"{len(all_videos)} videos used in total!")
    
    
    # get image size
    with av.open(vid_args[0][0]) as container:
        in_stream = container.streams.video[0]
        ih, iw = in_stream.height, in_stream.width
    
    
    # dump images
    img_compressor = JpegXl(level=compression_level, numthreads=1)
    for cam_id in range(n_cameras):
        name = f'camera{cam_id}_rgb'
        _ = out_replay_buffer.data.require_dataset(
            name=name,
            shape=(out_replay_buffer['robot0_eef_pos'].shape[0],) + out_res + (3,),
            chunks=(1,) + out_res + (3,),
            compressor=img_compressor,
            dtype=np.uint8
        )

    def video_to_zarr(replay_buffer, mp4_path, tasks):
        pkl_path = os.path.join(os.path.dirname(mp4_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        resize_tf = get_image_transform(
            in_res=(iw, ih),
            out_res=out_res
        )
        tasks = sorted(tasks, key=lambda x: x['frame_start'])
        camera_idx = None
        for task in tasks:
            if camera_idx is None:
                camera_idx = task['camera_idx']
            else:
                assert camera_idx == task['camera_idx']
        name = f'camera{camera_idx}_rgb'
        img_array = replay_buffer.data[name]
        
        curr_task_idx = 0
        
        is_mirror = None
        if mirror_swap:
            ow, oh = out_res
            mirror_mask = np.ones((oh,ow,3),dtype=np.uint8)
            mirror_mask = draw_predefined_mask(
                mirror_mask, color=(0,0,0), mirror=True, gripper=False, finger=False)
            is_mirror = (mirror_mask[...,0] == 0)
        
        with av.open(mp4_path) as container:
            in_stream = container.streams.video[0]
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            buffer_idx = 0
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if curr_task_idx >= len(tasks):
                    # all tasks done
                    break
                
                if frame_idx < tasks[curr_task_idx]['frame_start']:
                    # current task not started
                    continue
                elif frame_idx < tasks[curr_task_idx]['frame_end']:
                    if frame_idx == tasks[curr_task_idx]['frame_start']:
                        buffer_idx = tasks[curr_task_idx]['buffer_start']
                    
                    # do current task
                    img = frame.to_ndarray(format='rgb24')

                    # inpaint tags
                    this_det = tag_detection_results[frame_idx]
                    all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                    for corners in all_corners:
                        img = inpaint_tag(img, corners)
                        
                    # mask out gripper
                    img = draw_predefined_mask(img, color=(0,0,0), 
                        mirror=no_mirror, gripper=True, finger=False)
                    # resize
                    if fisheye_converter is None:
                        img = resize_tf(img)
                    else:
                        img = fisheye_converter.forward(img)
                        
                    # handle mirror swap
                    if mirror_swap:
                        img[is_mirror] = img[:,::-1,:][is_mirror]
                        
                    # compress image
                    img_array[buffer_idx] = img
                    buffer_idx += 1
                    
                    if (frame_idx + 1) == tasks[curr_task_idx]['frame_end']:
                        # current task done, advance
                        curr_task_idx += 1
                else:
                    assert False
                    
    with tqdm(total=len(vid_args)) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            for mp4_path, tasks in vid_args:
                if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))

                futures.add(executor.submit(video_to_zarr, 
                    out_replay_buffer, mp4_path, tasks))

            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print([x.result() for x in completed])

    # dump to disk
    print(f"Saving ReplayBuffer to {output}")
    with zarr.ZipStore(output, mode='w') as zip_store:
        out_replay_buffer.save_to_store(
            store=zip_store
        )
    print(f"Done! {len(all_videos)} videos used in total!")

# %%
if __name__ == "__main__":
    main()
