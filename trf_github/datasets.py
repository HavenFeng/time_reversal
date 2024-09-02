import os
import glob
import torch
import numpy as np
from diffusers.utils import load_image, export_to_video, export_to_gif

def get_frame_dataset(dataset_folder, filter=None):
    video_frame_selected = [

    'video_frames/DJI_clip1/frame_00000.jpg',
    'video_frames/DJI_clip1/frame_00057.jpg',

    'video_frames/dolomite_clip3/frame_00000.jpg',
    'video_frames/dolomite_clip3/frame_00100.jpg',

    'video_frames/re10k_008_gt/frame_00000.jpg',
    'video_frames/re10k_008_gt/frame_00012.jpg',

    'video_frames/tiffany_clip2/frame_00000.jpg',
    'video_frames/tiffany_clip2/frame_00032.jpg',

    'gym_motion_2024_frames/arm_clip1/frame_00018.jpg',
    'gym_motion_2024_frames/arm_clip1/frame_00060.jpg',

    'gym_motion_2024_frames/squat_clip3/frame_00026.jpg',
    'gym_motion_2024_frames/squat_clip3/frame_00060.jpg',

    'gym_motion_2024_frames/squat_clip1/frame_00000.jpg',
    'gym_motion_2024_frames/squat_clip1/frame_00049.jpg',

    'gym_motion_2024_frames/arm_clip1/frame_00180.jpg',
    'gym_motion_2024_frames/arm_clip1/frame_00205.jpg',

    'gym_motion_2024_frames/back_clip1/frame_00100.jpg',
    'gym_motion_2024_frames/back_clip1/frame_00130.jpg',

    ]
    # import ipdb; ipdb.set_trace()
    video_frame_selected = [dataset_folder + video_frame for video_frame in video_frame_selected]
    if filter is not None:
        video_frame_selected = [video_frame for video_frame in video_frame_selected if filter in video_frame]
    video_frames = np.array(video_frame_selected).reshape(-1, 2)
    video_frames_reverse = video_frames[:, ::-1]
    video_frames = np.vstack((video_frames, video_frames_reverse))
    return video_frames


def get_multiview_dataset(dataset_folder, filter=None):
    video_frame_selected = [

    'Multiview_data/campanile/0305.jpeg',
    'Multiview_data/campanile/0298.jpeg',
    
    'Multiview_data/campanile/0118.jpeg',
    'Multiview_data/campanile/0114.jpeg',

    'Multiview_data/DTU/scan65/000027.png',
    'Multiview_data/DTU/scan65/000032.png',

    'Multiview_data/mipnerf360_lite/garden/frame_00006.JPG',
    'Multiview_data/mipnerf360_lite/garden/frame_00104.JPG',

    'Multiview_data/mipnerf360_lite/room/frame_00010.JPG',
    'Multiview_data/mipnerf360_lite/room/frame_00018.JPG',

    'Multiview_data/LLFF/room/DJI_20200226_143942_875.png',
    'Multiview_data/LLFF/room/DJI_20200226_143929_618.png',

    ]
    if filter is not None:
        video_frame_selected = [dataset_folder + video_frame for video_frame in video_frame_selected if filter in video_frame]
    video_frames = np.array(video_frame_selected).reshape(-1, 2)
    video_frames_reverse = video_frames[:, ::-1]
    video_frames = np.vstack((video_frames, video_frames_reverse))

    return video_frames

def get_loop_dataset(dataset_folder, filter=None):
    video_frame_selected = sorted(glob.glob('{}img2loop/*/*'.format(dataset_folder)))
    if filter is not None:
        video_frame_selected = [video_frame for video_frame in video_frame_selected if filter in video_frame]
    images_np = np.array(video_frame_selected)
    video_frames = np.hstack((images_np.reshape(-1,1), images_np.reshape(-1,1)))

    return video_frames


def img_preprocess(img_path, orig_aspect=None):
    if orig_aspect is None:
        image = load_image(img_path)
        w, h = image.size
            # return image
        if w/h > 16/9:
            height = h
            width = int(height * 16 / 9)
        else:
            width = w
            height = int(width * 9 / 16)
        #center crop the image with width and height
        image = image.crop(((w - width) // 2, (h - height) // 2, (w + width) // 2, (h + height) // 2))
        image = image.resize((1024, 576))
        return image
    else:
        image = load_image(img_path)
        w, h = image.size
            # return image
        if w/h > 1:
            height = h
            width = height
        else:
            width = w
            height = width
        #center crop the image with width and height
        image = image.crop(((w - width) // 2, (h - height) // 2, (w + width) // 2, (h + height) // 2))
        image = image.resize((768, 768))
        return image
    
if __name__ == '__main__':

    output_folder = '/is/cluster/work/hfeng/datasets/video_dynamic_dataset/misc_crop'


    # for sample_folder in testing_samples:
    sample_folder = '/is/cluster/work/hfeng/datasets/video_dynamic_dataset/misc'
    sample_images = sorted(glob.glob(os.path.join(sample_folder, '*')))
    for sample_image in sample_images:
        output_path = output_folder + '/' + sample_image.split('/')[-1]
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        img = img_preprocess(sample_image)
        img.save(output_path)


