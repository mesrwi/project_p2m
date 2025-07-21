import os
import json
import cv2
import argparse

from tqdm import tqdm
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

import sys
sys.path.append('tools/hrnet')
from config import cfg
from config import update_config
from core.inference import get_multi_stage_outputs
from core.inference import aggregate_results
from core.nms import pose_nms
from core.match import match_pose_to_heatmap
from utils.transforms import resize_align_multi_scale
from utils.transforms import get_final_preds
from utils.transforms import get_multi_scale_size
import models

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

# 전처리 함수
def resize_with_padding(image, target_size=512, pad_color=(0,0,0)):
    h, w, _ = image.shape
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)

    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                borderType=cv2.BORDER_CONSTANT,
                                value=pad_color)
    return padded, scale, (left, top)

def get_pose_estimation_prediction(cfg, model, image, vis_thre, transforms):
    # size at scale 1.0
    base_size, center, scale = get_multi_scale_size(
        image, cfg.DATASET.INPUT_SIZE, 1.0, 1.0
    )

    with torch.no_grad():
        heatmap_sum = 0
        poses = []

        for scale in sorted(cfg.TEST.SCALE_FACTOR, reverse=True):
            image_resized, center, scale_resized = resize_align_multi_scale(
                image, cfg.DATASET.INPUT_SIZE, scale, 1.0
            )

            image_resized = transforms(image_resized)
            image_resized = image_resized.unsqueeze(0).to(device)

            heatmap, posemap = get_multi_stage_outputs(
                cfg, model, image_resized, cfg.TEST.FLIP_TEST
            )
            heatmap_sum, poses = aggregate_results(
                cfg, heatmap_sum, poses, heatmap, posemap, scale
            )
        
        heatmap_avg = heatmap_sum/len(cfg.TEST.SCALE_FACTOR)
        poses, scores = pose_nms(cfg, heatmap_avg, poses)

        if len(scores) == 0:
            return []
        else:
            if cfg.TEST.MATCH_HMP:
                poses = match_pose_to_heatmap(cfg, poses, heatmap_avg)

            final_poses = get_final_preds(
                poses, center, scale_resized, base_size
            )

        final_results = []
        for i in range(len(scores)):
            if scores[i] > vis_thre:
                final_results.append(final_poses[i])

        if len(final_results) == 0:
            return []

    return final_results

def prepare_output_dirs(outputDir):
    frame_dir = os.path.join(outputDir, 'frames')
    os.makedirs(frame_dir, exist_ok=True)
    
    return outputDir

def parse_args():
    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--cfg', type=str, default='tools/hrnet/inference_demo_coco.yaml')
    parser.add_argument('--visthre', type=float, default=0.1)
    
    args = parser.parse_args()
    
    return args

def main(input_vid, duration, vid_size, fps, target_size, target_fps):
    # transformation
    pose_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
    
    args = parse_args()
    # args.videoFile = filepath
    
    update_config(cfg, args)
    # pose_dir = prepare_output_dirs(args.outputDir)
    csv_output_rows = []
    pose2d = []
    
    # HRNet for pose 로드
    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    
    # 사전학습한 HRNet의 가중치 로드
    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        if device == torch.cuda.is_available():
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, weights_only=True), strict=False)
        else:
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, weights_only=True, map_location=device), strict=False)
    else:
        raise ValueError('expected model defined in config at TEST.MODEL_FILE')
    
    pose_model.to(device)
    pose_model.eval()
    
    skip_frame_cnt = round(fps / target_fps)
    count = 0
    pose_seq = []
    with tqdm(total=int(duration*target_fps), ncols=100, desc="Processing frames") as pbar:
        while input_vid.isOpened():
            ret, image_bgr = input_vid.read()
            count += 1
            
            if not ret:
                break
            if count % skip_frame_cnt != 0:
                continue
            
            # 프레임 리사이즈+패딩
            processed_frame, scale, offset = resize_with_padding(image_bgr, target_size)
            
            image_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            image_pose = image_rgb.copy()
            
            # Clone 1 image for debugging purpose
            image_debug = processed_frame.copy()
            
            pose_preds = get_pose_estimation_prediction(
                cfg, pose_model, image_pose, args.visthre, transforms=pose_transform)
            
            pose_seq.append(pose_preds[0])
            
            pbar.update(1)
    input_vid.release()
    
    return pose_seq


if __name__ == '__main__':
    main()