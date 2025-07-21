# 입력 영상 로드 및 처리: opencv
# 스켈레톤 정보 추출: DEKR, MotionBERT
# 모델에 입력할 수 있는 형태로 리턴: 실증을 위한 인퍼런스 시에는 굳이 저장할 필요 없음

'''
mp4 -> feeder.py -> Inputs(...)
'''
from typing import Dict
import os
import cv2
import numpy as np

import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample
from torchvision.transforms import Compose, Lambda
from transformers import CLIPProcessor

import tools.hrnet.run
import tools.motionbert.inference as motionbert

class ResizedPad(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.pad_color = (0, 0, 0)
    
    def __call__(self, vid):
        c, t, h, w = vid.shape
        scale = self.output_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        
        pad_w, pad_h = self.output_size - new_w, self.output_size - new_h
        top, bot, left, right = pad_h // 2, pad_h - (pad_h // 2), pad_w // 2, pad_w - (pad_w // 2)
        
        result = []
        for ind in range(t):
            frame = vid[:, ind, :, :].permute(1, 2, 0).numpy()
            resized_frame = cv2.resize(frame, (new_w, new_h))
            padded_frame = cv2.copyMakeBorder(resized_frame, top, bot, left, right,
                                            borderType=cv2.BORDER_CONSTANT, value=self.pad_color)
            
            result.append(torch.from_numpy(padded_frame).permute(2, 0, 1))
        
        return torch.stack(result, dim=1)

def CLIP_feeder(filepath: str, target_fps):
    input_vid = EncodedVideo.from_path(filepath)
    duration = int(input_vid.duration)
    transform = ApplyTransformToKey(
        key="video",
        transform=Compose(
            [
                UniformTemporalSubsample(duration*target_fps),
                Lambda(lambda x: x/255.0),
                ResizedPad(512)
            ]
        )
    )
    rgb_inputs = transform(input_vid.get_clip(start_sec=0, end_sec=8))['video'] # rgb_inputs: [C, T', H', W']
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_inputs = processor(images=rgb_inputs.permute(1, 0, 2, 3), do_rescale=False, return_tensors="pt", padding=True)
    
    return clip_inputs

def pose_feeder(filepath: str, target_size, target_fps):
    # Loading a video
    input_vid = cv2.VideoCapture(filepath)
    fps = input_vid.get(cv2.CAP_PROP_FPS)
    if fps < target_fps:
        raise ValueError('desired target fps is ' + str(target_fps) + ' but video fps is ' + str(fps))
    duration = int(input_vid.get(cv2.CAP_PROP_FRAME_COUNT)) // fps # sec
    
    vid_size = (input_vid.get(cv2.CAP_PROP_FRAME_WIDTH), input_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pose2d = np.stack(tools.hrnet.run.main(input_vid, duration, vid_size, fps, target_size=512, target_fps=10))
    pose3d = motionbert.main(pose2d, vid_size)
    
    return pose2d, pose3d

def input_feeder(filepath: str, target_size: int, target_fps: int) -> Dict:
    input_dict = {}
    print('Processing RGB inputs...')
    input_dict['CLIP'] = CLIP_feeder(filepath, target_fps)['pixel_values']
    print('Complete!')
    
    print('Extracting pose skeleton...')
    pose2d, pose3d = pose_feeder(filepath, target_size, target_fps)
    input_dict['pose2d'] = pose2d
    input_dict['pose3d'] = pose3d
    print('Complete!')
    
    return input_dict

if __name__=="__main__":
    filepath = '/data2/Pose2Muscle/test/231121_A/ground_heavy/00014/clip.mp4'
    input_dict = input_feeder(filepath, target_size=512, target_fps=10)