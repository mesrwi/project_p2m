import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from tools.pose2muscle.CLIP_LoRA_temporal import CLIPLoRATemporalModel
from feeder import input_feeder

def signed_angle_from_b_to_a(a, b):
    # 정규화
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    
    # dot과 cross
    dot = np.dot(b_norm, a_norm)
    cross = b_norm[0]*a_norm[1] - b_norm[1]*a_norm[0]  # b → a 방향

    # 부호 각도 계산
    angle_rad = np.arctan2(cross, dot)
    angle_deg = np.degrees(angle_rad)
    
    # 시계 방향이면 +, 반시계 방향이면 -로 뒤집기
    return -angle_deg

def predict_shoulder_degree(pose2d):
    theta_list = []
    for kpts in pose2d:
        right_shoulder = kpts[6]
        right_elbow = kpts[8]
        right_hip = kpts[12]
        right_ear = kpts[4]
        
        v_shoulder = np.array(right_elbow[:2]) - np.array(right_shoulder[:2])
        v_body = np.array(right_hip[:2]) - np.array(right_ear[:2])
        
        theta_list.append(signed_angle_from_b_to_a(v_shoulder, v_body))
    
    avg_degree = sum(theta_list) / len(theta_list)
    print("평균 어깨 각도:", round(avg_degree, 2))
    
    # 시각화화
    plt.plot(theta_list, label="Calculated shoulder degrees")
    plt.xlabel("Time Step")
    plt.ylabel("Degree")
    plt.title(f"Shoulder Degrees (Average: {round(avg_degree, 2)})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"degrees_plot.png")
    plt.cla()
    print(">> Saved plot to degrees_plot.png")
    
    # CSV로 저장
    np.savetxt(f"shoulder_degrees.csv", np.array(theta_list), delimiter=",")
    print(">> Saved prediction to shoulder_degrees.csv")

def visualization(pose3d):
    bones = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]] # MotionBERT; Human3.6M

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=12., azim=80)

    def update(frame):
        ax.clear()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        frame_data = pose3d[frame]
        
        for bone in bones:
            joint1, joint2 = bone
            ax.plot([-frame_data[joint1, 0], -frame_data[joint2, 0]],
                    [-frame_data[joint1, 2], -frame_data[joint2, 2]],
                    [-frame_data[joint1, 1], -frame_data[joint2, 1]], 'ro-', markersize=3)

    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=pose3d.shape[0])
    ani.save('pose3d.mp4',fps=10)

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
    
    predict_shoulder_degree(input_dict['pose2d'])
    visualization(input_dict['pose3d'])
    
    model = CLIPLoRATemporalModel.load_from_checkpoint('tools/pose2muscle/P2M_CLIP_LoRA_temporal-epoch=189-val_smape=0.66.ckpt', args=args)
    model = model.eval()
    
    x_CLIP = input_dict['CLIP'].unsqueeze(0).to(model.device)
    x_text = ['Inference']
    
    preds, image_embeds, text_embeds = model(x_CLIP, x_text)
    plt.plot(preds.squeeze().detach().cpu().numpy()[:, 2])
    plt.savefig('res.png')
    
    return preds

if __name__=='__main__':
    filepath = "original_clip_1.mp4" # '/data2/Pose2Muscle/test/231121_A/ground_heavy/00014/clip.mp4'
    preds = main(filepath)