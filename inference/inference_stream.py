#%%
from configs_parser import load_cfg
from lstr import LSTR, LSTRStream
import torch
import numpy as np
import os
from feature_models.combined_flow_extractor import CombinedFlowModel
from feature_models.resnet import resnet50
from PIL import Image
from torchvision import transforms
from tqdm import trange
from utils import perframe_average_precision
import time
import argparse

CONFIG_FILE = 'lstr_long_256_work_8_kinetics_1x.yaml'
GPU = '0'
OPTS = None
# Create agrument parser with video_name as argument
parser = argparse.ArgumentParser()
parser.add_argument('--video_name', type=str, 
                    default="01.폭행(assult)__insidedoor_07__412-4__412-4_cam02_assault01_place09_night_spring_resized")
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()


device = 'cuda:'+args.gpu if torch.cuda.is_available() else 'cpu'
checkpoint = "/data/common/abb_project/processed_data/epoch-25.pth"
#device = 'cpu'
cfg = load_cfg(CONFIG_FILE, OPTS, GPU)
model: LSTRStream = LSTRStream(cfg)
model.load_state_dict(torch.load(checkpoint)['model_state_dict'])
model.to(device)
model.eval()


video_name = args.video_name
target_perframe = np.load("/data/common/abb_project/features/target_perframe/"+video_name+".npy")

frames_path = "/data/common/abb_project/frames/"+video_name
frames_list = sorted(os.listdir(frames_path))
rgb_model = resnet50("../checkpoints/pretrained/resnet50.pth").to(device)
flow_model = CombinedFlowModel().to(device)
rgb_model.eval()
flow_model.eval()


MEAN=[123.675/255, 116.28/255, 103.53/255]
STD=[58.395/255, 57.12/255, 57.375/255]
EVERY_N_FRAMES = 6
SHORT_SIDE = 512
transform = transforms.Compose([
            transforms.Resize(SHORT_SIDE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]) 

# Collect scores and targets

long_memory_length = cfg.MODEL.LSTR.LONG_MEMORY_LENGTH
long_memory_sample_rate = cfg.MODEL.LSTR.LONG_MEMORY_SAMPLE_RATE
long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
work_memory_length = cfg.MODEL.LSTR.WORK_MEMORY_LENGTH
work_memory_sample_rate = cfg.MODEL.LSTR.WORK_MEMORY_SAMPLE_RATE
work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES

# %%
 # %%
current_long_memory_index = long_memory_num_samples - 1
pred_scores = []
gt_targets = []
# Save every 6 images
images = []
start_time = time.time()
with torch.no_grad():
    #for current_iter in trange(target_perframe.shape[0]):
    for frame_idx in trange(0, len(frames_list)):
        
        # Wait until 3 images are collected
        current_iter = frame_idx // EVERY_N_FRAMES

        # Take 0, 2, 5 frames
        if frame_idx % EVERY_N_FRAMES == 0 or frame_idx % EVERY_N_FRAMES == 2 or frame_idx % EVERY_N_FRAMES == 5:
            frame = Image.open(os.path.join(frames_path, frames_list[frame_idx]))
            frame = transform(frame).to(device)
            images.append(frame)
        else:
            continue    
        if len(images) < 3:
            continue
        
        frame_rgb_features = rgb_model(images[1].unsqueeze(0))
        frame_flow_features = flow_model(torch.cat([images[0], images[2]]).unsqueeze(0))
        frame_flow_features.squeeze_(2, 3)
        # Empty the images list
        images = []


        # Get work memory
        if current_iter == 0:
            work_visual_inputs = frame_rgb_features.repeat(work_memory_length, 1)
            work_motion_inputs = frame_flow_features.repeat(work_memory_length, 1)
        else:
            work_visual_inputs = torch.cat([work_visual_inputs[1:], frame_rgb_features])
            work_motion_inputs = torch.cat([work_motion_inputs[1:], frame_flow_features])

        long_end = current_iter - 1
        # If it is the first frame fill the long memory cache with the first frame repeated
        # After that each frame replaces the oldest frame in the long memory cache
        # Since the long memory cache is filled with the first frame, it will stay until queue is full
        if long_end == -1:
            long_visual_inputs = torch.cat([work_visual_inputs[[0]]]*long_memory_num_samples)
            long_motion_inputs = torch.cat([work_motion_inputs[[0]]]*long_memory_num_samples)
        # Every long_memory_sample_rate frames, the long memory cache is updated
        # Update logic is inside the model inference function
        elif long_end % long_memory_sample_rate == 0:
            long_visual_inputs = work_visual_inputs[[0]] # Get the oldest frame in the work memory
            long_motion_inputs = work_motion_inputs[[0]]
            if long_end > 0:
                current_long_memory_index -= 1
        else: # If the long memory cache is not updated, pass None
            long_visual_inputs = None 
            long_motion_inputs = None

        # Put -inf in the memory key padding mask for the frames that are not in the long memory cache
        # It won't matter after long memory cache is filled
        memory_key_padding_mask = torch.zeros(long_memory_num_samples, dtype=torch.float32).to(device)
        if current_long_memory_index > 0:
            memory_key_padding_mask[:current_long_memory_index] = float('-inf')

        # I didn't want to squeeze and unsqueeze every time, so I just added a dimension afterwards
        if long_visual_inputs is None:
            in_long_visual_inputs = None
            in_long_motion_inputs = None
        else:
            in_long_visual_inputs = long_visual_inputs.unsqueeze(0)
            in_long_motion_inputs = long_motion_inputs.unsqueeze(0)

        in_work_visual_inputs = work_visual_inputs.unsqueeze(0)
        in_work_motion_inputs = work_motion_inputs.unsqueeze(0)
        in_memory_key_padding_mask = memory_key_padding_mask.unsqueeze(0)

        # Get the prediction for work_memory_length frames
        score = model.stream_inference(
                        in_long_visual_inputs,
                        in_long_motion_inputs,
                        in_work_visual_inputs,
                        in_work_motion_inputs,
                        in_memory_key_padding_mask)[0]

        score = score.softmax(dim=-1).cpu().numpy()


        
        gt_targets.append(list(target_perframe[current_iter]))
        pred_scores.append(list(score[-1]))
# %%
result = perframe_average_precision(gt_targets, pred_scores, ["normal", "assault", "wander", "trespass"], -1, "AP", None)
print(f"mAP: {result['mean_AP']:.5f}")
end_time = time.time()
print(f"Running time: {end_time - start_time:.3f} seconds")