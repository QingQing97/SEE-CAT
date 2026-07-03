import os
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import pickle
import cv2
from openpose.src.body import Body
from tqdm import tqdm
import logging
import sys
import argparse


# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 确认设备是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 MTCNN 和 FaceNet (InceptionResnetV1) 并转移到 GPU
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=40, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 初始化 OpenPose 模型
body_estimator = Body('openpose/model/body_pose_model.pth')


# 获取面部编码
def get_face_encoding(image):
    # 在 GPU 上运行 MTCNN 和 FaceNet
    face = mtcnn(image)
    if face is not None:
        face = face.to(device)  # 确保面部图像在GPU上
        encoding = resnet(face.unsqueeze(0))
        return encoding.detach().cpu().numpy().flatten()  # 转移到 CPU 并转换为 numpy
    else:
        return None  # 如果没有检测到脸，返回 None


# 获取姿态编码，移除低置信度的关键点
def get_pose_encoding(image_path, confidence_threshold):
    image = cv2.imread(image_path)
    candidate, subset = body_estimator(image)

    # 初始化 18 个关键点的 (x, y) 坐标
    keypoints = np.zeros((18, 2))

    if len(candidate) > 0:
        # 遍历检测到的关键点并根据置信度进行过滤
        valid_points = 0
        for point in candidate:
            keypoint_id = int(point[3])  # 通过 point[3] 获取节点ID
            if keypoint_id < 18 and point[2] >= confidence_threshold:  # 只保留置信度超过阈值的关键点
                keypoints[keypoint_id] = point[:2]  # 提取 (x, y) 坐标并保存在 keypoint_id 对应的位置
                valid_points += 1

        if valid_points > 0:
            # 返回没有置信度的 keypoints，作为编码
            pose_encoding = keypoints.flatten()
            return pose_encoding
        else:
            return None  # 如果没有有效的关键点，返回 None
    else:
        return None  # 如果没有检测到姿态，返回 None

# 处理图像目录并保存嵌入
def process_and_save_embeddings(video_dir, save_dir, pose_threshold):
    # 创建保存嵌入的目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 遍历每帧
    for frame_name in tqdm(os.listdir(video_dir), desc="Processing video folders"):
        image_path = os.path.join(video_dir, frame_name)
        
        # 检查文件是否为图像
        if not frame_name.endswith('.jpg'):
            continue

        # 加载图像并转为RGB
        image = Image.open(image_path).convert("RGB")

        # 获取面部编码
        face_encoding = get_face_encoding(image)

        # 如果没有检测到脸，跳过该图片
        if face_encoding is None:
            logging.info(f"No face detected for {image_path}, skipping.")
            continue

        # 获取姿态编码
        pose_encoding = get_pose_encoding(image_path, pose_threshold)

        # # 如果没有检测到有效姿态，跳过该图片
        # if pose_encoding is None:
        #     logging.info(f"No valid pose detected for {image_path}, skipping.")
        #     continue

        if pose_encoding is None:
            pose_encoding = np.zeros(36)

        # 拼接面部编码和姿态编码
        combined_encoding = np.concatenate([face_encoding, pose_encoding])

        # 保存嵌入为 pkl 文件
        save_path = os.path.join(save_dir, frame_name.replace('.jpg', '.pkl'))
        with open(save_path, 'wb') as f:
            pickle.dump(combined_encoding, f)
            
        # logging.info(f"Saved embedding for {image_path} at {save_path}")


# 主函数
# tbb要对应模型
# 可设置poseCT
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train and evaluate the VisualEncoder model.")

    # Add arguments for configurable input parameters
    parser.add_argument('--dataset', type=str, default='MELD', help='MELD or IEMOCAP')
    parser.add_argument('--tbb', type=float, default=0.7, help='bounding box confidence threshold')
    parser.add_argument('--type', type=str, default='train', help='train/dev/test')
    parser.add_argument('--poseCT', type=float, default=0.6, help='pose confidence threshold')

    # Parse the arguments
    args = parser.parse_args()
    # train_2,dev,test
    speaker_image_dir = f'/nfs/users/gaoqingqing/Denoise/Seven/MidResults/{args.dataset}_SpeakerImages/ViT-L/14/3btd/confidence_threshold_{args.tbb}/{args.type}'
    speaker_embedding_dir = f'MidResults/{args.dataset}_SpeakerEmbeddings_548/tbb_{args.tbb}_concise_refine_{args.poseCT}/{args.type}_new/'
    process_and_save_embeddings(speaker_image_dir, speaker_embedding_dir, args.poseCT)

    logging.info("Embedding processing complete.")
    print(f"tbb={args.tbb},poseCT={args.poseCT},type={args.type}")
