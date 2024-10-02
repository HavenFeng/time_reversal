import os
import glob
import numpy as np

def get_dataset(dataset_folder, data_type='frame', filter_keyword=None):
    """
    General function to retrieve datasets based on the data_type.

    Parameters:
        dataset_folder (str): The root directory of the dataset.
        data_type (str): Type of dataset to retrieve. Options are 'frame', 'multiview', 'loop'.
        filter_keyword (str, optional): A keyword to filter the dataset.

    Returns:
        np.ndarray: An array of image pairs.
    """
    if data_type == 'loop':
        # For loop dataset, pairs are the same image
        image_paths = sorted(glob.glob(os.path.join(dataset_folder, 'img2loop', '*', '*')))
        if filter_keyword:
            image_paths = [path for path in image_paths if filter_keyword in path]
        video_frames = np.array([[path, path] for path in image_paths])
    else:
        # For 'frame' and 'multiview' datasets
        if data_type == 'frame':
            video_frame_selected = [
                'video_frames/dolomite_clip3/frame_00000.jpg',
                'video_frames/dolomite_clip3/frame_00100.jpg',

                'gym_motion_2024_frames/arm_clip1/frame_00018.jpg',
                'gym_motion_2024_frames/arm_clip1/frame_00060.jpg',
            ]
        elif data_type == 'multiview':
            video_frame_selected = [
                'Multiview_data/mipnerf360_lite/garden/frame_00006.JPG',
                'Multiview_data/mipnerf360_lite/garden/frame_00104.JPG',
            ]
        else:
            raise ValueError(f"Unsupported data_type: {data_type}")

        # Prepend dataset_folder to paths
        video_frame_selected = [os.path.join(dataset_folder, path) for path in video_frame_selected]
        if filter_keyword:
            video_frame_selected = [path for path in video_frame_selected if filter_keyword in path]

        # Reshape into pairs
        video_frames = np.array(video_frame_selected).reshape(-1, 2)

    return video_frames
