import cv2
import numpy as np
import os
import torch
import tqdm


def preprocess_pose_video(video_path, target_frames=50, target_size=(100, 100)):
    """
    1. 均勻採樣 50 幀
    2. 縮放至 100x100
    3. 歸一化與格式轉換
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"無法打開影片: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 均勻採樣索引計算
    if total_frames >= target_frames:
        indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    else:
        # 如果影片不足 50 幀則進行重複填充
        indices = np.arange(total_frames)
        print(f"警告: {video_path} 幀數不足")

    frames = []
    current_idx = 0
    selected_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if selected_count < target_frames and current_idx == indices[selected_count]:
            # 1. 調整解析度至 100x100
            frame = cv2.resize(frame, target_size)
            # 2. 轉換為 RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 3. 歸一化 [0, 1]
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
            selected_count += 1

        current_idx += 1

    cap.release()

    # 處理幀數不足的情況
    while len(frames) < target_frames:
        frames.append(frames[-1] if frames else np.zeros((target_size[0], target_size[1], 3)))

    # 返回維度: (Frames, H, W, C) -> (50, 100, 100, 3)
    return np.array(frames)


def batch_convert_to_npy(input_dir, output_dir):
    """
    批量處理資料夾內的所有骨架影片並保存為 npy
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_list = [f for f in os.listdir(input_dir) if f.endswith(('.avi', '.mp4'))]

    for video_name in tqdm.tqdm(video_list):
        video_path = os.path.join(input_dir, video_name)
        processed_data = preprocess_pose_video(video_path)

        if processed_data is not None:
            save_name = os.path.splitext(video_name)[0] + ".npy"
            np.save(os.path.join(output_dir, save_name), processed_data)


if __name__ == "__main__":
    # 修改為你的路徑
    SOURCE_POSE_DIR = "../data/RWF2000/noFights_pose"
    OUTPUT_NPY_DIR = "../data/RWF2000/noFights_pose_npy"

    batch_convert_to_npy(SOURCE_POSE_DIR, OUTPUT_NPY_DIR)