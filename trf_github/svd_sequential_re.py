import os
import glob
import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
from pipeline_stable_video_diffusion_re import StableVideoDiffusionPipeline_Custom

from diffusers.utils import load_image, export_to_video, export_to_gif
from unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from scheduling_euler_discrete_resampling import EulerDiscreteScheduler
from datasets import get_frame_dataset, get_loop_dataset, get_multiview_dataset, get_gym_dataset, get_misc_samples
root_dir = os.path.dirname(os.path.abspath(__file__))


model_card = "stabilityai/stable-video-diffusion-img2vid-xt"

original_pipe = StableVideoDiffusionPipeline.from_pretrained(
    model_card, torch_dtype=torch.float16, variant="fp16"
)
unet_custom = UNetSpatioTemporalConditionModel.from_pretrained(model_card, subfolder="unet", torch_dtype=torch.float16, variant="fp16").to('cuda')
scheduler_custom = EulerDiscreteScheduler.from_pretrained(model_card, subfolder="scheduler", torch_dtype=torch.float16, variant="fp16")
pipe = StableVideoDiffusionPipeline_Custom(
    vae = original_pipe.vae,
    image_encoder = original_pipe.image_encoder,
    unet = unet_custom,
    scheduler = scheduler_custom,
    feature_extractor = original_pipe.feature_extractor,
)

pipe.enable_model_cpu_offload()


def img_preprocess(img_path, mode='crop', orig_aspect=None):
    image = load_image(img_path)
    w, h = image.size
    
    if orig_aspect is None:
        aspect_ratio = 16 / 9
    else:
        aspect_ratio = orig_aspect
    
    if mode == 'crop':
        # Crop the image to the specified aspect ratio
        if w / h > aspect_ratio:
            height = h
            width = int(height * aspect_ratio)
        else:
            width = w
            height = int(width / aspect_ratio)
        
        # Center crop the image with width and height
        left = (w - width) // 2
        top = (h - height) // 2
        right = (w + width) // 2
        bottom = (h + height) // 2
        image = image.crop((left, top, right, bottom))
        image = image.resize((1024, 576))
    
    elif mode == 'padding':
        # Pad the image to the specified aspect ratio
        new_w = w
        new_h = h
        
        if w / h < aspect_ratio:
            new_w = int(h * aspect_ratio)
        else:
            new_h = int(w / aspect_ratio)
        
        new_image = Image.new("RGB", (new_w, new_h), (0, 0, 0))
        new_image.paste(image, ((new_w - w) // 2, (new_h - h) // 2))
        image = new_image.resize((1024, 576))
    
    return image

data_type = 'multiview' # video_frames
dataset_folder = '{}/test_data/'.format(root_dir)
output_folder = '/{}/output/{}_exp/'.format(root_dir, data_type)
fps_value = 17
motion_id = 10
random_seed = 42
generator = torch.manual_seed(random_seed)
num_inference_steps=50
jump_n_sample = 3

jump_length = 5
repeat_step_ratio = 0.7
noise_scale_ratio = 1.
model_folder = output_folder + '{}_fps{}_id{}_s-num{}_re{}_{}_{}_{}_std_1p_l_crop'.format(model_card.split('/')[-1],fps_value, motion_id, num_inference_steps, int((1-repeat_step_ratio) * num_inference_steps), jump_length, jump_n_sample, noise_scale_ratio)
if data_type == 'image2loop':
    video_frames = get_loop_dataset(dataset_folder)
elif data_type == 'video_frames':
    video_frames = get_frame_dataset(dataset_folder, filter='video_frames') 
elif data_type == 'multiview':
    video_frames = get_multiview_dataset(dataset_folder)

for image_pair in video_frames:
    start_frame = img_preprocess(image_pair[0])
    end_frame = img_preprocess(image_pair[1])

    if data_type == 'image2loop':
        frame_folder = model_folder + '/' + image_pair[0].split('/')[-2] + '_' + image_pair[0].split('/')[-1][:-4]
    elif data_type == 'video_frames':
        frame_folder = model_folder + '/' + image_pair[0].split('/')[-2] + '_' + image_pair[0].split('_')[-1][:-4] + '_' + image_pair[1].split('_')[-1][:-4]
    elif data_type == 'multiview':
        frame_folder = model_folder + '/' + image_pair[0].split('/')[-2] + '_' + image_pair[0].split('/')[-1].split('_')[-1][:-4] + '_' + image_pair[1].split('/')[-1].split('_')[-1][:-4]
    
    if not os.path.exists(frame_folder):
        os.makedirs(frame_folder)
    video_file = frame_folder + '.mp4'
    img_w, img_h = start_frame.size
    frames = pipe(start_frame, end_frame, height=img_h, width=img_w, num_frames=25, num_inference_steps=num_inference_steps, fps=fps_value, jump_length=jump_length, jump_n_sample=jump_n_sample, repeat_step_ratio=repeat_step_ratio, noise_scale_ratio=noise_scale_ratio, decode_chunk_size=8, motion_bucket_id=motion_id, generator=generator).frames[0]
    
    for i, frame in enumerate(frames):
        frame.save(f'{frame_folder}/{i}.png')
        
    export_to_video(frames, video_file, fps=fps_value)
