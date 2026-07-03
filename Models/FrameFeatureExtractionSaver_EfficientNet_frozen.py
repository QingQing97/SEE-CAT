import os
import torch
import torch.nn as nn
import pickle
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import time
from datetime import datetime


class FrameFeatureExtractor(nn.Module):
    def __init__(self, freeze_level='full'):
        super(FrameFeatureExtractor, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b0').to(self.device)

        # 添加全局平均池化层 -> 降维
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # 冻结控制
        if freeze_level == 'full':
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
        elif freeze_level == 'partial':
            for i in range(10):
                for param in self.feature_extractor._blocks[i].parameters():
                    param.requires_grad = False
    
    def extract_frame_features(self, image_paths):
        """批量提取帧特征"""
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
            images.append(image)
        
        images = torch.stack(images).to(self.device)
        
        with torch.no_grad():
            features = self.feature_extractor.extract_features(images)  # [B, 1280, 7, 7]
            features = self.global_avg_pool(features)  # [B, 1280, 1, 1]
            features = features.view(features.size(0), -1)  # [B, 1280]
        
        return features  # [B, 62720]


class FrameFeatureExtractionSaver(nn.Module):
    def __init__(self, dataset="IEMOCAP"):
        super(FrameFeatureExtractionSaver, self).__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = dataset

        
        print(f"Initializing FrameFeatureExtractionSaver")
        print(f"Using device: {self.device}")
        
        # 初始化特征提取器
        self.frame_feature_extractor = FrameFeatureExtractor()
        
        # 特征保存的根目录
        self.frame_embedding_dirs = {
            'train': f"MidResults/{self.dataset}_FrameEmbeddings/train",
            'dev': f"MidResults/{self.dataset}_FrameEmbeddings/dev",
            'test': f"MidResults/{self.dataset}_FrameEmbeddings/test"
        }
        
        # 创建目录
        for dir_path in self.frame_embedding_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.to(self.device)
    
    def get_all_video_folders(self, root_path):
        """递归获取所有包含图片文件的文件夹
        
        Args:
            root_path: 根目录路径
        
        Returns:
            video_folders: 所有包含图片的文件夹路径列表
        """
        video_folders = []
        
        if not os.path.exists(root_path):
            print(f"Warning: Path does not exist: {root_path}")
            return video_folders
        
        for item in os.listdir(root_path):
            item_path = os.path.join(root_path, item)
            
            if os.path.isdir(item_path):
                # 检查当前文件夹是否包含图片
                has_images = False
                try:
                    for f in os.listdir(item_path):
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
                            has_images = True
                            break
                except PermissionError:
                    continue
                
                if has_images:
                    # 如果当前文件夹包含图片，就把它作为一个视频文件夹
                    video_folders.append(item_path)
                else:
                    # 否则继续递归查找子文件夹
                    sub_folders = self.get_all_video_folders(item_path)
                    video_folders.extend(sub_folders)
        
        return video_folders
    
    def save_frame_features(self, video_folder_path, split):
        """提取并保存视频中所有帧的特征
        
        Args:
            video_folder_path: 视频文件夹路径（包含帧图像的文件夹）
            split: 'train', 'dev', 或 'test'
        
        Returns:
            saved_paths: 保存的特征文件路径列表
        """
        # 获取所有图片文件
        frame_paths = []
        try:
            frame_files = os.listdir(video_folder_path)
        except PermissionError:
            print(f"Permission denied: {video_folder_path}")
            return []
        
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        for frame_file in frame_files:
            frame_path = os.path.join(video_folder_path, frame_file)
            if (not os.path.basename(frame_path).startswith('.') and 
                frame_file.lower().endswith(valid_extensions)):
                frame_paths.append(frame_path)
        
        if len(frame_paths) == 0:
            print(f"Warning: No valid frames found in {video_folder_path}")
            return []
        
        # 按文件名排序，保证顺序一致
        frame_paths.sort()
        
        # 批量提取所有帧的特征
        all_frame_features = self.frame_feature_extractor.extract_frame_features(frame_paths)
        
        # 将特征转换到 CPU 并保存
        saved_paths = []
        # 使用文件夹名作为视频名（取最后一级目录名）
        video_name = os.path.basename(video_folder_path)
        save_dir = self.frame_embedding_dirs[split]
        
        for idx, frame_path in enumerate(frame_paths):
            # 获取帧特征并移到 CPU
            frame_feature = all_frame_features[idx].cpu().numpy()
            
            # 生成保存文件名
            frame_filename = os.path.basename(frame_path)
            frame_name_without_ext = os.path.splitext(frame_filename)[0]
            save_filename = f"{video_name}_{frame_name_without_ext}.pkl"
            save_path = os.path.join(save_dir, save_filename)
            
            # 保存特征
            with open(save_path, 'wb') as f:
                pickle.dump(frame_feature, f)
            
            saved_paths.append(save_path)
        
        print(f"  Processed {len(frame_paths)} frames.")
        return saved_paths
    
    def process_split(self, split_root, split):
        """处理一个 split 下的所有视频文件夹（支持多层嵌套）
        
        Args:
            split_root: split 的根目录路径，如 "E:/Data/IEMOCAP.DProcess/test"
            split: 'train', 'dev', 或 'test'
        
        Returns:
            all_saved_paths: 所有保存的特征文件路径
        """
        all_saved_paths = []
        
        if not os.path.exists(split_root):
            print(f"Error: Path {split_root} does not exist!")
            return all_saved_paths
        
        # 递归获取所有包含图片的文件夹
        video_folders = self.get_all_video_folders(split_root)
        
        print(f"\n{'='*60}")
        print(f"Processing {split.upper()} split")
        print(f"Root path: {split_root}")
        print(f"Found {len(video_folders)} video folders")
        print(f"{'='*60}")
        
        for idx, video_path in enumerate(video_folders):
            # 显示相对路径，便于阅读
            relative_path = os.path.relpath(video_path, split_root)
            print(f"\n[{idx+1}/{len(video_folders)}] Processing: {relative_path}")
            
            try:
                saved_paths = self.save_frame_features(video_path, split)
                all_saved_paths.extend(saved_paths)
            except Exception as e:
                print(f"  Error: {e}")
                continue
        
        print(f"\n{'='*60}")
        print(f"Completed {split.upper()} split: saved {len(all_saved_paths)} features")
        print(f"{'='*60}\n")
        
        return all_saved_paths
    
    def process_all_splits(self, data_root):
        """处理 DProcess 下所有 split（train/dev/test）
        
        Args:
            data_root: IEMOCAP.DProcess 的根目录路径
                     如 "E:/Data/IEMOCAP.DProcess"
        
        Returns:
            all_results: 字典，包含每个 split 的保存路径
        """
        splits = ['train', 'dev', 'test']
        all_results = {}
        
        for split in splits:
            split_path = os.path.join(data_root, split)
            if os.path.exists(split_path):
                saved_paths = self.process_split(split_path, split)
                all_results[split] = saved_paths
            else:
                print(f"Warning: {split_path} does not exist, skipping...")
        
        return all_results
    
    def forward(self, video_folder_path, split):
        """forward 方法，用于兼容原有的调用方式"""
        return self.save_frame_features(video_folder_path, split)


if __name__ == "__main__":
    
    # 记录整个程序开始时间
    program_start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*60)
    print(f"程序开始时间: {start_datetime}")
    print("="*60)
    
    # 创建特征提取保存器
    dataset = "MELD"
    extractor = FrameFeatureExtractionSaver(dataset=dataset)
    
    # ============================================
    # 方式1: 处理所有 split（train/dev/test）
    # ============================================
    data_root = f"E:/Data/{dataset}.DProcess"  # 改成你的路径
    
    all_results = {}
    
    if os.path.exists(data_root):
        all_results = extractor.process_all_splits(data_root)
    else:
        print(f"Data root not found: {data_root}")
        print("Please update the 'data_root' variable to your correct path.")
    
    # ============================================
    # 方式2: 只处理单个 split
    # ============================================
    # test_split_path = "E:/Data/IEMOCAP.DProcess/test"
    # if os.path.exists(test_split_pat
    #     saved_paths = extractor.process_split(test_split_path, split="test")
    #     all_results = {"test": saved_paths}
    
    # ============================================
    # 方式3: 处理单个视频文件夹
    # ============================================
    # video_path = "E:/Data/IEMOCAP.DProcess/test/Ses05F_impro01/Ses05F_impro01_utt0"
    # if os.path.exists(video_path):
    #     saved_paths = extractor.save_frame_features(video_path, split="test")
    #     all_results = {"test": saved_paths}
    
    # ========== 所有输出统一放在最后 ==========
    # 整个程序结束时间
    program_end_time = time.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    total_elapsed = program_end_time - program_start_time
    
    print("\n" + "="*60)
    print("运行时间统计")
    print("="*60)
    print(f"程序结束时间: {end_datetime}")
    print(f"总运行时间: {total_elapsed:.2f} 秒 ({total_elapsed/60:.2f} 分钟)")
    print("="*60)
    
    # 打印统计信息
    if all_results:
        print("\n" + "="*60)
        print("特征保存统计")
        print("="*60)
        total = 0
        for split, paths in all_results.items():
            count = len(paths)
            total += count
            print(f"{split.upper()}: {count} features saved")
        print(f"总计: {total} features saved")
        print("="*60)
        
        # 输出两个 pkl 示例
        print("\n" + "="*60)
        print("保存文件示例")
        print("="*60)
        
        # 收集示例文件（优先从 train 取，如果没有则从其他 split）
        example_files = []
        for split in ['train', 'dev', 'test']:
            if split in all_results and len(all_results[split]) > 0:
                for path in all_results[split][:2]:
                    example_files.append((split, path))
                if len(example_files) >= 2:
                    break
        
        # 打印示例
        for idx, (split, file_path) in enumerate(example_files[:2], 1):
            # 获取文件大小
            file_size = os.path.getsize(file_path) / 1024  # KB
            
            # 获取特征维度（加载示例检查）
            try:
                with open(file_path, 'rb') as f:
                    feature = pickle.load(f)
                feature_shape = feature.shape
            except:
                feature_shape = "N/A"
            
            print(f"\n示例 {idx} [{split.upper()}]:")
            print(f"  路径: {file_path}")
            print(f"  大小: {file_size:.2f} KB")
            print(f"  维度: {feature_shape}")
        
        print("\n" + "="*60)
    else:
        print("\n未保存任何特征文件，请检查数据路径是否正确。")