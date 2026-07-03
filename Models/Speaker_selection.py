import os
import torch
import cv2
from PIL import Image
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
import clip
import time
from datetime import datetime

class SpeakerSelector:
    """说话人选择器：基于 RTMDet + CLIP 从视频帧中提取说话人图像"""
    
    def __init__(self, device=None, t_bb=0.85):
        """
        初始化说话人选择器
        
        Args:
            device: 运行设备，默认自动检测
            t_bb: 边界框置信度阈值
        """
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.t_bb = t_bb
        
        # OD 模型配置（需要用户指定路径）
        self.od_config_file = None
        self.od_checkpoint_file = None
        self.od_model = None
        
        # CLIP 模型
        self.clip_model_name = "ViT-L/14"
        self.clip_model = None
        self.clip_preprocess = None
        
        # 文本描述
        self.text_descriptions = [
            "a person speaking",
            "a person listening",
            "a background person",
        ]
        self.text_tokens = None
    
    def init_od_model(self, config_file, checkpoint_file):
        """初始化目标检测模型（RTMDet）"""
        self.od_config_file = config_file
        self.od_checkpoint_file = checkpoint_file
        self.od_model = init_detector(config_file, checkpoint_file, device=self.device)
        print(f"OD model initialized: {config_file}")
    
    def init_clip_model(self, model_name="ViT-L/14"):
        """初始化 CLIP 模型"""
        self.clip_model_name = model_name
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.text_tokens = clip.tokenize(self.text_descriptions).to(self.device)
        print(f"CLIP model initialized: {model_name}")
    
    def detect_persons(self, image_path):
        """检测图像中的人体边界框"""
        result = inference_detector(self.od_model, image_path)
        image = cv2.imread(image_path)
        instances = result.pred_instances
        
        # 获取人体检测框（COCO 中 person 类别的索引是 0）
        human_boxes = instances[instances.labels == 0].bboxes.cpu().numpy()
        scores = instances[instances.labels == 0].scores.cpu().numpy()
        
        # 过滤低置信度边界框
        human_boxes = human_boxes[scores >= self.t_bb]
        scores = scores[scores >= self.t_bb]
        
        return image, human_boxes, scores
    
    def classify_person(self, cropped_image):
        """对裁剪的人体图像进行分类（说话人/听者/背景）"""
        # BGR 转 RGB
        cropped_image_rgb = cropped_image[..., ::-1]
        cropped_image_pil = Image.fromarray(cropped_image_rgb)
        
        # CLIP 预处理
        image_input = self.clip_preprocess(cropped_image_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits_per_image, _ = self.clip_model(image_input, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            state_idx = probs.argmax()
            confidence = probs.max()
            
            return state_idx, confidence, probs
    
    def select_speaker(self, image_path):
        """
        从单张图像中选择说话人
        
        Returns:
            speaker_image: PIL Image 或 None（无说话人）
            confidence: 置信度
        """
        image, human_boxes, scores = self.detect_persons(image_path)
        
        if len(human_boxes) == 0:
            return None, 0.0
        
        best_confidence = 0
        best_speaker_image = None
        best_box = None
        
        for idx, box in enumerate(human_boxes):
            x1, y1, x2, y2 = map(int, box)
            cropped_image = image[y1:y2, x1:x2]
            
            state_idx, confidence, _ = self.classify_person(cropped_image)
            
            # 只考虑说话人（state_idx == 0）且置信度高于当前最佳
            if state_idx == 0 and confidence > best_confidence:
                best_confidence = confidence
                cropped_rgb = cropped_image[..., ::-1]
                best_speaker_image = Image.fromarray(cropped_rgb)
                best_box = box
        
        return best_speaker_image, best_confidence
    
    def process_dataset(self, base_image_dir, output_base_dir, dtype="train"):
        """
        处理整个数据集，提取所有说话人图像
        
        Args:
            base_image_dir: 输入图像根目录
            output_base_dir: 输出说话人图像根目录
            dtype: 数据集类型（train/dev/test）
        """
        speaker_output_dir = os.path.join(output_base_dir, dtype)
        os.makedirs(speaker_output_dir, exist_ok=True)
        
        print(f"Processing {dtype} set...")
        print(f"Input: {base_image_dir}")
        print(f"Output: {speaker_output_dir}")
        
        stats = {
            "total_frames": 0,
            "frames_with_speaker": 0,
            "frames_without_speaker": 0
        }
        
        for sub_video_folder in tqdm(os.listdir(base_image_dir), desc="Processing subjects"):
            sub_video_path = os.path.join(base_image_dir, sub_video_folder)
            
            if not os.path.isdir(sub_video_path):
                continue
            
            for video_folder in tqdm(os.listdir(sub_video_path), desc=f"Processing {sub_video_folder}", leave=False):
                video_path = os.path.join(sub_video_path, video_folder)
                
                if not os.path.isdir(video_path):
                    continue
                
                image_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                
                for image_file in image_files:
                    stats["total_frames"] += 1
                    image_path = os.path.join(video_path, image_file)
                    
                    speaker_image, confidence = self.select_speaker(image_path)
                    
                    if speaker_image is not None:
                        stats["frames_with_speaker"] += 1
                        output_filename = f"{video_folder}_{image_file[:-4]}.jpg"
                        output_path = os.path.join(speaker_output_dir, output_filename)
                        speaker_image.save(output_path)
                    else:
                        stats["frames_without_speaker"] += 1
        
        # 打印统计信息
        print("\n" + "="*60)
        print(f"Processing complete for {dtype}")
        print("="*60)
        print(f"Total frames: {stats['total_frames']}")
        print(f"Frames with speaker: {stats['frames_with_speaker']}")
        print(f"Frames without speaker: {stats['frames_without_speaker']}")
        print(f"Output directory: {speaker_output_dir}")
        print("="*60)
        
        return stats



if __name__ == "__main__":

    # 记录整个程序开始时间
    program_start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*60)
    print(f"程序开始时间: {start_datetime}")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_bb = 0.85  # 边界框置信度阈值
    dtype = "train"  # train / dev / test
    dataset = "IEMOCAP"

    print(device, t_bb, dtype, dataset)
    
    # OD 模型路径（请根据实际情况修改）
    od_config_file = 'E:/OthersCode/mmdetection-main/configs/rtmdet/rtmdet_l_swin_b_p6_4xb16-100e_coco.py'
    od_checkpoint_file = 'E:/OthersCode/mmdetection-main/models/rtmdet_l_swin_b_p6_4xb16-100e_coco-a1486b6f.pth'
    
    # 输入输出路径
    base_image_dir = f'E:/Data/{dataset}.DProcess/{dtype}'
    speaker_output_base_dir = f'MidResults/{dataset}_SpeakerImages'
    
    # ==================== 初始化选择器 ====================
    selector = SpeakerSelector(device=device, t_bb=t_bb)
    
    # 初始化模型
    selector.init_od_model(od_config_file, od_checkpoint_file)
    selector.init_clip_model()  
    
    # 处理数据集
    stats = selector.process_dataset(
        base_image_dir=base_image_dir,
        output_base_dir=speaker_output_base_dir,
        dtype=dtype
    )
    
    print(f"\nDone! Speaker images saved to {speaker_output_base_dir}/{dtype}")
    print(f"Text descriptions: {selector.text_descriptions}")


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