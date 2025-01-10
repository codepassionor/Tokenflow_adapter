import os
import cv2
from utils import load_config

def extract_frames(video_path, output_folder, height, width):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # 获取帧的尺寸
            frame_height, frame_width = frame.shape[:2]
            # 计算裁剪的起始点
            start_x = frame_width // 2 - width // 2
            start_y = frame_height // 2 - height // 2
            # 如果计算出的起始点小于0，则设置为0
            start_x = max(start_x, 0)
            start_y = max(start_y, 0)
            # 裁剪帧
            frame = frame[start_y:start_y+height, start_x:start_x+width]
            cv2.imwrite(os.path.join(output_folder, f"frame{count:04d}.png"), frame)
            count += 1
        else:
            break
    cap.release()

for i in range(20):
    config = load_config(i)
    real_images_folder = f"data/" + config["work_dir"]
    # 从配置中读取高度和宽度
    frame_height = config['height']
    frame_width = config['width']
    extract_frames(config['input_path'], real_images_folder, frame_height, frame_width)
