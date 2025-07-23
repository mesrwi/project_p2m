import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from tools.pose2muscle.CLIP_LoRA_temporal import CLIPLoRATemporalModel
from feeder import CLIP_feeder

import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader

bins = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # 수정 가능

def annotation(y):
    return np.digitize(y, bins)

def tolerant_accuracy(y_true, y_pred, tolerance=1):
    return (np.abs(y_true - y_pred) <= tolerance).sum() / len(y_true)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, help="Path to input video file")
    parser.add_argument("--model", default='CLIPLoRATemporal')
    parser.add_argument("--temporal_range", default=9)
    parser.add_argument("--d_model", default=128)
    parser.add_argument("--fusion_method", default="concat")
    parser.add_argument("--lr", default=1e-4)
    args = parser.parse_args()
    
    return args

def main(args):
    model = CLIPLoRATemporalModel.load_from_checkpoint('tools/pose2muscle/P2M_CLIP_LoRA_temporal-epoch=189-val_smape=0.66.ckpt', args=args)
    model = model.eval()
    
    if os.path.isfile(args.filepath):
        clip_inputs = torch.load(args.filepath)
        x_CLIP = clip_inputs.unsqueeze(0).to(model.device)
        x_text = ['Inference']
        
    else:
        filepath = os.path.dirname(args.filepath) + '/clip.mp4'
        clip_inputs = CLIP_feeder(filepath, target_fps=10, save=True)
        
        x_CLIP = clip_inputs['pixel_values'].unsqueeze(0).to(model.device)
        x_text = ['Inference']
    
    preds, image_embeds, text_embeds = model(x_CLIP, x_text)
    # plt.plot(preds.squeeze(0).detach().cpu().numpy()[:, 2])
    # plt.xlabel("Time Step")
    # plt.ylabel(f"Prediction (Channel 2)")
    # plt.title("Predicted Signal")
    # plt.savefig('res.png')
    
    return preds

class TestDataset(Dataset):
    def __init__(self, target):
        super().__init__()
        self.filedirs = target
        # self.clip_inputs = []
        # self.gts = []
        # for filedir in self.filedirs:
        #     self.clip_inputs.append(torch.load(filedir+'/clip_inputs.pt'))
        #     self.gts.append(torch.from_numpy(np.load(filedir+'/emgvalues.npy'))[:, 1:])
    
    def __len__(self):
        return len(self.filedirs)
    
    def __getitem__(self, idx):
        clip_input = torch.load(self.filedirs[idx] + '/clip_inputs.pt', weights_only=True)
        gt = torch.from_numpy(np.load(self.filedirs[idx] + '/emgvalues.npy'))[:, 1:]
        return clip_input, gt
    
def main_test(args):
    print('Loading model...')
    model = CLIPLoRATemporalModel.load_from_checkpoint('tools/pose2muscle/P2M_CLIP_LoRA_temporal-epoch=189-val_smape=0.66.ckpt', args=args, weights_only=True)
    model = model.eval()
    print('Complete!')
    
    filedirs = glob.glob('/data2/Pose2Muscle/test/*/ground_*/*')
    test_dataset = TestDataset(filedirs)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)
    
    preds_list = []
    gts_list = []
    with torch.no_grad():
        for item in tqdm(test_loader):
            x_CLIP, gts = item
            gts_list.append(gts)
            x_CLIP = x_CLIP.to(model.device)
            x_text = ['Inference']
            
            preds, image_embeds, text_embeds = model(x_CLIP, x_text)
            preds_list.append(preds.cpu())
    
    return preds_list, gts_list
    
if __name__=='__main__':
    args = parse_args()
    preds_list, gts_list = main_test(args)
    preds = np.concatenate(preds_list).reshape(-1, 8)
    gts = np.concatenate(gts_list).reshape(-1, 8)
    
    muscles = ["척추기립근", "상부승모근", "상완이두근", "천지굴근", "지신근", "대퇴이두근", "외측광근", "외복사근"]
    plt.figure(figsize=(20, 10))
    mae_list = []
    for i in range(8):
        plt.subplot(4, 2, i+1)
        plt.plot(gts[:, i]/100)
        plt.plot(preds[:, i])
        plt.savefig('test_res.png')
        mae = np.mean(np.abs(gts[:, i]/100 - preds[:, i]))
        mae_list.append((1 - mae)*100)
        print(f"'{muscles[i]}'에 대한 예측 정확도: {(1 - mae)*100:.2f} %")
        
    print(f"전체 평균: {sum(mae_list) / len(mae_list):.2f}")
    
    # filedirs = glob.glob('/data2/Pose2Muscle/test/*/move_*/*')
    # args = parse_args()
    # preds_list = []
    # gts_list = []
    # for filedir in tqdm(filedirs):
    #     args.filepath = filedir+'/clip_inputs.pt'        
    #     preds_list.append(main(args).squeeze(0).detach().cpu().numpy())
    #     gts_list.append(np.load(os.path.dirname(args.filepath)+'/emgvalues.npy')[:, 1:])
    
    # gts = np.concatenate(gts_list)
    # preds = np.concatenate(preds_list)
    
    # annot_func = np.vectorize(annotation)
    # plt.figure(figsize=(20, 10))
    # for i in range(7):
    #     y_test = annot_func(gts[:, i]/100)
    #     y_pred = annot_func(preds[:, i])
    #     plt.subplot(4, 2, i+1)
    #     plt.plot(y_test)
    #     plt.plot(y_pred)
    #     plt.savefig('test_res.png')
        
    #     acc_exact = (y_test == y_pred).sum() / len(y_test)
    #     # acc_tolerant = tolerant_accuracy(y_test, y_pred, tolerance=1)
    #     acc_tolerant = tolerant_accuracy(gts[:, i]/100, preds[:, i], tolerance=0.05)
    #     mae = np.mean(np.abs(gts[:, i]/100 - preds[:, i]))

    #     print(f"Muscle {i}: Exact Acc = {acc_exact:.3f}, Tolerant Acc = {acc_tolerant:.3f}, MAE: {mae:.4f}, MAE_Acc: {1 - mae:.4f}")