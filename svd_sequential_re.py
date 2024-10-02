import os, sys

import tqdm._tqdm
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from diffusers.utils import load_image, export_to_video
from pipeline_stable_video_diffusion_re import StableVideoDiffusionPipeline_Custom
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from scheduling_euler_discrete_resampling import EulerDiscreteScheduler
from datasets import get_dataset
import tqdm

def img_preprocess(img_path, mode='crop', orig_aspect=None):
    image = load_image(img_path)
    w, h = image.size
    aspect_ratio = orig_aspect if orig_aspect else 16 / 9

    if mode == 'crop':
        # Crop the image to the specified aspect ratio
        if w / h > aspect_ratio:
            height = h
            width = int(height * aspect_ratio)
        else:
            width = w
            height = int(width / aspect_ratio)
        left = (w - width) // 2
        top = (h - height) // 2
        image = image.crop((left, top, left + width, top + height))
        image = image.resize((1024, 576))
    elif mode == 'padding':
        # Pad the image to the specified aspect ratio
        new_w = int(h * aspect_ratio) if w / h < aspect_ratio else w
        new_h = int(w / aspect_ratio) if w / h >= aspect_ratio else h
        new_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        new_image.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
        image = new_image.resize((1024, 576))
    return image

def time_reversal_fusion(model_card, start_frame, end_frame, num_inference_steps, fps_value, jump_n_sample, jump_length, repeat_step_ratio, noise_scale_ratio, motion_id, generator):
    # Load model components
    original_pipe = StableVideoDiffusionPipeline.from_pretrained(
        model_card, torch_dtype=torch.float16, variant="fp16"
    )
    unet_custom = UNetSpatioTemporalConditionModel.from_pretrained(
        model_card, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
    ).to('cuda')
    scheduler_custom = EulerDiscreteScheduler.from_pretrained(
        model_card, subfolder="scheduler", torch_dtype=torch.float16, variant="fp16"
    )
    # Generate frames
    pipe = StableVideoDiffusionPipeline_Custom(
        vae=original_pipe.vae,
        image_encoder=original_pipe.image_encoder,
        unet=unet_custom,
        scheduler=scheduler_custom,
        feature_extractor=original_pipe.feature_extractor,
    )
    pipe.enable_model_cpu_offload()
    frames = pipe(
        start_frame,
        end_frame,
        height=start_frame.height,
        width=start_frame.width,
        num_frames=25,
        num_inference_steps=num_inference_steps,
        fps=fps_value,
        jump_length=jump_length,
        jump_n_sample=jump_n_sample,
        repeat_step_ratio=repeat_step_ratio,
        noise_scale_ratio=noise_scale_ratio,
        decode_chunk_size=8,
        motion_bucket_id=motion_id,
        generator=generator,
    ).frames[0]

    return frames

def main(data_type):
    # Configuration
    root_dir = os.path.dirname(os.path.abspath(__file__))
    # data_type = 'multiview'  # Options: 'image2loop', 'video_frames', 'multiview'
    dataset_folder = os.path.join(root_dir, 'test_data')
    output_folder = os.path.join(root_dir, 'output', f'{data_type}_exp')
    os.makedirs(output_folder, exist_ok=True)

    # Model parameters
    model_card = "stabilityai/stable-video-diffusion-img2vid-xt"
    fps_value = 7
    motion_id = 127
    random_seed = 42
    num_inference_steps = 50
    jump_n_sample = 2
    jump_length = 5
    repeat_step_ratio = 0.8
    noise_scale_ratio = 1.0
    generator = torch.manual_seed(random_seed)
    
    # Load data-specific settings
    if data_type == 'image2loop':
        video_frames = get_dataset(dataset_folder, data_type='loop')
        fps_value = 7
        motion_id = 127
        jump_n_sample = 2
        jump_length = 5
        repeat_step_ratio = 0.8
    elif data_type == 'video_frames':
        video_frames = get_dataset(dataset_folder, data_type='frame', filter_keyword='video_frames')
        fps_value = 7
        motion_id = 127
        jump_n_sample = 2
        jump_length = 5
        repeat_step_ratio = 0.8
        noise_scale_ratio = .95

    elif data_type == 'gym_motion':
        video_frames = get_dataset(dataset_folder, data_type='frame', filter_keyword='gym_motion')
        fps_value = 17
        motion_id = 10
        jump_n_sample = 2
        jump_length = 5
        repeat_step_ratio = 0.8
    elif data_type == 'multiview':
        video_frames = get_dataset(dataset_folder, data_type='multiview')
        fps_value = 7
        motion_id = 127
        jump_n_sample = 2
        jump_length = 5
        repeat_step_ratio = 0.8
        noise_scale_ratio = 1

    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    # Model directory
    re_steps = int((1 - repeat_step_ratio) * num_inference_steps)
    model_folder_name = f"{model_card.split('/')[-1]}_fps{fps_value}_id{motion_id}_s-num{num_inference_steps}_re{re_steps}_{jump_length}_{jump_n_sample}_{noise_scale_ratio}"
    model_folder = os.path.join(output_folder, model_folder_name)
    os.makedirs(model_folder, exist_ok=True)

    # Process each image pair
    for idx in tqdm.tqdm(range(len(video_frames))):
        image_pair = video_frames[idx]
        start_frame = img_preprocess(image_pair[0])
        end_frame = img_preprocess(image_pair[1])

        # Generate frame folder name
        base_name_start = os.path.splitext(os.path.basename(image_pair[0]))[0]
        base_name_end = os.path.splitext(os.path.basename(image_pair[1]))[0]
        dir_name = os.path.basename(os.path.dirname(image_pair[0]))

        if data_type == 'image2loop':
            frame_folder_name = f"{dir_name}_{base_name_start}"
        else:
            frame_folder_name = f"{dir_name}_{base_name_start}_{base_name_end}"

        frame_folder = os.path.join(model_folder, frame_folder_name)
        os.makedirs(frame_folder, exist_ok=True)
        video_file = f"{frame_folder}.mp4"

        frames = time_reversal_fusion(
            model_card, start_frame, end_frame, num_inference_steps, fps_value, jump_n_sample, jump_length, repeat_step_ratio, noise_scale_ratio, motion_id, generator
        )

        # Save frames
        for i, frame in enumerate(frames):
            frame.save(os.path.join(frame_folder, f'{i}.png'))

        # Export to video
        export_to_video(frames, video_file, fps=fps_value)

if __name__ == '__main__':
    #different input flags for different datasets
    args = sys.argv[1:]
    data_type = args[0]

    main(data_type)
