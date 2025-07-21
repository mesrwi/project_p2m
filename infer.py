import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from tools.pose2muscle.CLIP_LoRA_temporal import CLIPLoRATemporalModel
from feeder import input_feeder

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='CLIPLoRATemporal')
    parser.add_argument("--temporal_range", default=9)
    parser.add_argument("--d_model", default=128)
    parser.add_argument("--fusion_method", default="concat")
    parser.add_argument("--lr", default=1e-4)
    args = parser.parse_args()
    
    return args

def main(filepath):
    args = parse_args()
    input_dict = input_feeder(filepath, target_size=512, target_fps=10)
    
    model = CLIPLoRATemporalModel.load_from_checkpoint('tools/pose2muscle/P2M_CLIP_LoRA_temporal-epoch=189-val_smape=0.66.ckpt', args=args)
    model = model.eval()
    
    x_CLIP = input_dict['CLIP'].unsqueeze(0).to(model.device)
    x_text = ['Inference']
    
    preds, image_embeds, text_embeds = model(x_CLIP, x_text)
    plt.plot(preds.squeeze().detach().cpu().numpy()[:, 2])
    plt.savefig('res.png')
    
    return preds

if __name__=='__main__':
    filepath = "/home/tako/mesrwi/Pose2Muscle/examples/original_clip_1.mp4" # '/data2/Pose2Muscle/test/231121_A/ground_heavy/00014/clip.mp4'
    preds = main(filepath)