import glob
import tqdm
import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys

# --- 配置 ---
DATASET = 'RWF2000'
BASE_DIR = Path(f'../data/{DATASET}') 
GAMMA = 0.67 
MODEL_PATH = '../models/yolo11x-pose.pt'

# 预计算 Gamma 表
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]).astype("uint8")

def ensure_dir(file_path):
    # 确保输出文件的父目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

def process_gamma(input_path, output_path):
    ensure_dir(output_path)
    
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"Error opening video: {input_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.LUT(frame, gamma_table)
        out.write(frame)

    cap.release()
    out.release()

def yolo_pose_estimation(model, input_path, output_path):
    ensure_dir(output_path)

    cap = cv2.VideoCapture(str(input_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False) # verbose=False 减少控制台刷屏
        annotated_frame = results[0].plot(img=np.zeros_like(frame), boxes=False, conf=False, labels=False)
        out.write(annotated_frame)

    cap.release()
    out.release()

if __name__ == '__main__':
    target_folders = ['fights_original', 'noFights_original']
    video_files = []
    
    for folder in target_folders:
        search_path = BASE_DIR / folder
        found = list(search_path.glob('*.avi'))
        video_files.extend(found)
    print(f"Found {len(video_files)} videos.")

    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model not found at {MODEL_PATH}.")
        sys.exit(1)
    else:
        print("Model founded.")
        model = YOLO(MODEL_PATH)

    for video_path in tqdm.tqdm(video_files):
        video_path = Path(video_path) # 转为 Path 对象
        
        # --- 步骤 1: Gamma 校正 ---
        current_input = video_path
        
        if GAMMA > 0:
            # 构建 Gamma 输出路径
            # 例如: fights_original -> fights_gamma
            parent_folder_name = video_path.parent.name
            new_folder_name = parent_folder_name.replace('original', 'gamma')
            gamma_output_path = video_path.parent.parent / new_folder_name / video_path.name
            
            process_gamma(current_input, gamma_output_path)
            
            # 更新下一步的输入为 Gamma 后的视频
            current_input = gamma_output_path
        
        # --- 步骤 2: 骨架识别 ---
        input_parent_name = current_input.parent.name
        
        # 构建骨架输出路径，将 'original' 或 'gamma' 替换为 'pose'
        if 'gamma' in input_parent_name:
            pose_folder_name = input_parent_name.replace('gamma', 'pose')
        elif 'original' in input_parent_name:
            pose_folder_name = input_parent_name.replace('original', 'pose')
        else:
            # 如果路径既没有 gamma 也没有 original，手动拼凑
            pose_folder_name = input_parent_name + "_pose"
        pose_output_path = current_input.parent.parent / pose_folder_name / current_input.name
        
        yolo_pose_estimation(
            model=model,
            input_path=current_input,
            output_path=pose_output_path
        )