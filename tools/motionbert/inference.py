import os
import math
import cv2
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from collections import OrderedDict
from tools.motionbert.utils.tools import *
from tools.motionbert.utils.learning import load_backbone
from tools.motionbert.utils.data import flip_data, crop_scale
# from tools.motionbert.data.dataset_coco import DetDataset

def coco2h36m(x):
    '''
        Input: x (T x V x C)
        COCO:                   H36M:
        0: nose,                0: root,
        1: left_eye,            1: right_hip,
        2: right_eye,           2: right_knee,
        3: left_ear,            3: right_ankle,
        4: right_ear,           4: left_hip,
        5: left_shoulder,       5: left_knee,
        6: right_shoulder,      6: left_ankle,
        7: left_elbow,          7: belly,
        8: right_elbow,         8: neck,
        9: left_wrist,          9: nose,
        10: right_wrist,        10: head,
        11: left_hip,           11: left_shoulder,
        12: right_hip,          12: left_elbow,
        13: left_knee,          13: left_wrist,
        14: right_knee,         14: right_shoulder,
        15: left_ankle,         15: right_elbow,
        16: right_ankle         16: right_wrist
    '''
    y = np.zeros(x.shape)
    y[:, 0, :] = (x[:, 11, :] + x[:, 12, :]) * 0.5
    y[:, 1, :] = x[:, 12, :]
    y[:, 2, :] = x[:, 14, :]
    y[:, 3, :] = x[:, 16, :]
    y[:, 4, :] = x[:, 11, :]
    y[:, 5, :] = x[:, 13, :]
    y[:, 6, :] = x[:, 15, :]
    y[:, 8, :] = (x[:, 5, :] + x[:, 6, :]) * 0.5
    y[:, 7, :] = (y[:, 0, :] + y[:, 8, :]) * 0.5
    y[:, 9, :] = x[:, 0, :]
    y[:, 10, :] = (x[:, 1, :] + x[:, 2, :]) * 0.5
    y[:, 11, :] = x[:, 5, :]
    y[:, 12, :] = x[:, 7, :]
    y[:, 13, :] = x[:, 9, :]
    y[:, 14, :] = x[:, 6, :]
    y[:, 15, :] = x[:, 8, :]
    y[:, 16, :] = x[:, 10, :]
    
    return y

class DetDataset(Dataset):
    def __init__(self, pose, clip_len, vid_size=None, scale_range=None, focus=None):
        self.pose = coco2h36m(pose)
        self.clip_len = clip_len
        
        if vid_size:
            w, h = vid_size
            scale = min(w, h) / 2.0
            self.pose[:, :, :2] = self.pose[:, :, :2] - np.array([w, h]) / 2.0
            self.pose[:, :, :2] = self.pose[:, :, :2] / scale
        if scale_range:
            self.pose = crop_scale(self.pose, scale_range)
        
        self.pose = self.pose.astype(np.float32)
    
    def __len__(self):
        return math.ceil(len(self.pose) / self.clip_len)
    
    def __getitem__(self, index):
        st = index * self.clip_len
        end = min((index + 1) * self.clip_len, len(self.pose))
        
        return self.pose[st: end]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='tools/motionbert/configs/MB_ft_h36m_global_lite.yaml', help='Path to the config file.')
    parser.add_argument('-e', '--evaluate', default='tools/motionbert/checkpoint/best_epoch.bin')
    parser.add_argument('-j', '--json_path', type=str, help='Detected 2D pose json path')
    parser.add_argument('-v', '--vid_path', type=str, help='video path')
    parser.add_argument('-o', '--out_path', type=str, help='output path')
    parser.add_argument('--pixel', action='store_true', help='align with pixel coordinates')
    parser.add_argument('--focus', type=int, default=None, help='target person id')
    parser.add_argument('--clip_len', type=int, default=243, help='clip length for network input')

    opts, _ = parser.parse_known_args()
    return opts

def main(pose2d, vid_size):
    opts = parse_args()
    # opts.out_path = '/home/tako/mesrwi/project_p2m/pipeline/res'
    args = get_config(opts.config)
    
    print('Loading checkpoint', opts.evaluate)
    model_backbone = load_backbone(args)
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()
        checkpoint = torch.load(opts.evaluate, map_location=lambda storage, loc: storage, weights_only=True)
        model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
    else:
        checkpoint = torch.load(opts.evaluate, map_location=torch.device('cpu'), weights_only=True)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_pos'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model_backbone.load_state_dict(new_state_dict, strict=True)

    model_pos = model_backbone
    model_pos.eval()
    test_loader_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 8,
        'pin_memory': True,
        'prefetch_factor': 4,
        'persistent_workers': True,
        'drop_last': False
    }
    
    # os.makedirs(opts.out_path, exist_ok=True)

    if opts.pixel:
        # Keep relative scale with pixel coornidates
        wild_dataset = DetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
    else:
        # Scale to [-1,1]
        # wild_dataset = DetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)
        wild_dataset = DetDataset(pose2d, clip_len=opts.clip_len, scale_range=[1, 1], focus=opts.focus)

    test_loader = DataLoader(wild_dataset, **test_loader_params)

    results_all = []
    with torch.no_grad():
        for batch_input in test_loader:
            N, T = batch_input.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf: # Input 2D keypoints without confidence score.
                batch_input = batch_input[:, :, :, :2]
            if args.flip: # flip data for augmentation
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # flip back
                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0 # ensemble
            else:
                predicted_3d_pos = model_pos(batch_input)
            
            if args.rootrel: # 모든 프레임의 root 좌표를 0으로 설정
                predicted_3d_pos[:, :, 0, :] = 0
            else: # 첫 프레임에서 root의 z좌표만 0으로 설정
                predicted_3d_pos[:, 0, 0, 2] = 0
                pass
            
            if args.gt_2d:
                predicted_3d_pos[..., :2] = batch_input[..., :2]
            
            results_all.append(predicted_3d_pos.cpu().numpy())
            
    results_all = np.hstack(results_all)
    results_all = np.concatenate(results_all)
    
    if opts.pixel:
        results_all = results_all * (min(vid_size) / 2.0)
        results_all[:, :, :2] = results_all[:, :, :2] + np.array(vid_size) / 2.0
        
    # filename = os.path.splitext(os.path.basename(opts.vid_path))
    # _, output_filename = os.path.split(opts.vid_path)
    # output_filename, _ = os.path.splitext(output_filename)
    # save_path = os.path.join(opts.out_path, output_filename+'.npy')
    # print('npy file saving...')
    # np.save(save_path, results_all)
    
    return results_all
