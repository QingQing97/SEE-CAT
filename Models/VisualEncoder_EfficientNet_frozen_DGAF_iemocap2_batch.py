import os
import torch
import torch.nn as nn
import pickle
import sys
import numpy as np
import random
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class CrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super(CrossAttentionFusion, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(emb_dim, num_heads)

    def forward(self, A, B):
        A_updated, _ = self.multihead_attn(A, B, B)
        return A_updated


class VisualEncoderTransRefine(nn.Module):
    def __init__(self, emb_dim=512, num_labels=7, tbb=0.85, poseCT=0.6, dropout_rate=0.3):
        super(VisualEncoderTransRefine, self).__init__()
        seed_everything(888)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.emb_dim = emb_dim
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate
                
        # 初始化各个模块
        self.cross_attention_fusion = CrossAttentionFusion(emb_dim=emb_dim, num_heads=8)
        
        self.gate_layer = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.Sigmoid()
        )
        
        # 线性映射层，将向量映射到同一维度
        self.fcA = nn.Linear(548, emb_dim)
        self.fcB = nn.Linear(1280, emb_dim)
        
        # Transformer Encoder
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_layers=6)
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_rate)
        
        # 线性映射层，将特征映射到标签空间
        self.fc = nn.Linear(emb_dim, num_labels)
        self.dataset = "IEMOCAP"
        
        # 帧表征路径，根据参数动态设置
        self.frame_embedding_dirs = {
            'train': f"MidResults/{self.dataset}_FrameEmbeddings/train/",
            'dev': f"MidResults/{self.dataset}_FrameEmbeddings/dev/",
            'test': f"MidResults/{self.dataset}_FrameEmbeddings/test/"
        }
        
        # 说话者表征路径，根据参数动态设置
        self.speaker_embedding_dirs = {
            'train': f"MidResults/{self.dataset}_SpeakerEmbeddings_548/tbb_{tbb}_concise_refine_{poseCT}/train_new2/",
            'dev': f"MidResults/{self.dataset}_SpeakerEmbeddings_548/tbb_{tbb}_concise_refine_{poseCT}/dev_new/",
            'test': f"MidResults/{self.dataset}_SpeakerEmbeddings_548/tbb_{tbb}_concise_refine_{poseCT}/test_new/"
        }
        
        self.to(self.device)
    
        
    def _get_speaker_embedding(self, frame_path, split):
        """获取说话者嵌入"""
        frame_filename = os.path.basename(frame_path)
        video_folder = os.path.basename(os.path.dirname(frame_path))
        
        speaker_embedding_dir = self.speaker_embedding_dirs[split]
        speaker_embedding_file = f"{video_folder}_{frame_filename.replace('.jpg', '.pkl')}"
        speaker_embedding_path = os.path.join(speaker_embedding_dir, speaker_embedding_file)
        
        if os.path.exists(speaker_embedding_path):
            with open(speaker_embedding_path, 'rb') as f:
                speaker_embedding = pickle.load(f)
            return torch.from_numpy(speaker_embedding).float().to(self.device)
        return None
    
    def _load_video_frames(self, video_folder_path, split):
        """加载单个视频的所有帧特征"""
        frame_files = [f for f in os.listdir(video_folder_path) 
                       if f.endswith('.jpg') and not f.startswith('.')]
        
        if len(frame_files) == 0:
            return None, None, None
        
        frame_features = []
        speaker_embeddings = []
        
        for frame_name in frame_files:
            frame_feature_path = os.path.join(
                self.frame_embedding_dirs[split],
                f"{os.path.basename(video_folder_path)}_{frame_name.replace('.jpg', '.pkl')}"
            )
            
            if not os.path.exists(frame_feature_path):
                continue
            
            with open(frame_feature_path, 'rb') as f:
                frame_feature_data = pickle.load(f)
            
            if isinstance(frame_feature_data, np.ndarray):
                frame_feature = torch.from_numpy(frame_feature_data).float()
            else:
                frame_feature = frame_feature_data.float()
            
            frame_features.append(frame_feature)
            
            frame_path = os.path.join(video_folder_path, frame_name)
            speaker_emb = self._get_speaker_embedding(frame_path, split)
            speaker_embeddings.append(speaker_emb)
        
        return frame_features, speaker_embeddings, frame_files
    
    def _process_single_video(self, video_folder_path, split):
        """处理单个视频"""
        frame_files = os.listdir(video_folder_path)
        
        if len(frame_files) == 0:
            zero_feature = torch.zeros(self.emb_dim).to(self.device)
            zero_logits = torch.zeros(self.num_labels).to(self.device)
            return zero_feature, zero_logits
        
        visual_features = []
        
        for frame_name in frame_files:
            if frame_name.startswith('.'):
                continue
                
            # 加载帧特征
            frame_feature_path = os.path.join(
                self.frame_embedding_dirs[split], 
                f"{os.path.basename(video_folder_path)}_{frame_name.replace('.jpg', '.pkl')}"
            )
            
            if not os.path.exists(frame_feature_path):
                continue
                
            with open(frame_feature_path, 'rb') as f:
                frame_feature_data = pickle.load(f)
            
            if isinstance(frame_feature_data, np.ndarray):
                frame_feature = torch.from_numpy(frame_feature_data).float().to(self.device)
            else:
                frame_feature = frame_feature_data.float().to(self.device)
            
            frame_feature = frame_feature.unsqueeze(0)  # [1, 1280]
            frame_feature = self.fcB(frame_feature)    # [1, emb_dim]
            
            # 获取说话者嵌入
            frame_path = os.path.join(video_folder_path, frame_name)
            speaker_embedding = self._get_speaker_embedding(frame_path, split)
            
            if speaker_embedding is not None:
                speaker_encoding = speaker_embedding.unsqueeze(0).to(self.device)  # [1, 548]
                speaker_encoding = self.fcA(speaker_encoding)  # [1, emb_dim]
                
                ca_fused = self.cross_attention_fusion(
                    speaker_encoding.unsqueeze(0),  # [1, 1, emb_dim]
                    frame_feature.unsqueeze(0)      # [1, 1, emb_dim]
                )
                gate = self.gate_layer(ca_fused)
                visual_feature = gate * ca_fused + (1 - gate) * speaker_encoding.unsqueeze(0)
                visual_feature = visual_feature.squeeze(0)  # [1, emb_dim]
            else:
                visual_feature = frame_feature.unsqueeze(0)  # [1, emb_dim]
            
            visual_features.append(visual_feature)  # 每个 [1, emb_dim]
        
        if len(visual_features) == 0:
            zero_feature = torch.zeros(self.emb_dim).to(self.device)
            zero_logits = torch.zeros(self.num_labels).to(self.device)
            return zero_feature, zero_logits
        
        # 堆叠并处理
        stacked_features = torch.cat(visual_features, dim=0)  # [num_frames, emb_dim]
        stacked_features = stacked_features.unsqueeze(0)      # [1, num_frames, emb_dim]
        
        transformer_output = self.transformer_encoder(stacked_features)  # [1, num_frames, emb_dim]
        
        if self.dropout:
            transformer_output = self.dropout(transformer_output)
        
        max_pooled_feature = torch.max(transformer_output, dim=1)[0]  # [1, emb_dim]
        logits = self.fc(max_pooled_feature)  # [1, num_labels]
        
        return max_pooled_feature.squeeze(0), logits.squeeze(0)
    
    def _process_batch(self, video_folder_paths, split):
        """批量处理多个视频"""
        # 收集所有视频的信息
        all_frame_features = []
        all_speaker_embeddings = []
        frame_counts = []
        
        for video_path in video_folder_paths:
            frame_features, speaker_embs, _ = self._load_video_frames(video_path, split)
            
            if frame_features is None or len(frame_features) == 0:
                frame_counts.append(0)
                continue
            
            frame_counts.append(len(frame_features))
            
            for ff in frame_features:
                ff_tensor = ff.to(self.device).unsqueeze(0)  # [1, 1280]
                ff_mapped = self.fcB(ff_tensor)             # [1, emb_dim]
                all_frame_features.append(ff_mapped)
            
            for se in speaker_embs:
                if se is not None:
                    se_tensor = se.to(self.device).unsqueeze(0)  # [1, 548]
                    se_mapped = self.fcA(se_tensor)             # [1, emb_dim]
                    all_speaker_embeddings.append(se_mapped)
                else:
                    all_speaker_embeddings.append(None)
        
        if len(all_frame_features) == 0:
            batch_size = len(video_folder_paths)
            zero_features = torch.zeros(batch_size, self.emb_dim).to(self.device)
            zero_logits = torch.zeros(batch_size, self.num_labels).to(self.device)
            return zero_features, zero_logits
        
        # 批量处理所有帧
        visual_features = []
        for i, (frame_feat, speaker_emb) in enumerate(zip(all_frame_features, all_speaker_embeddings)):
            if speaker_emb is not None:
                ca_fused = self.cross_attention_fusion(
                    speaker_emb.unsqueeze(0),  # [1, 1, emb_dim]
                    frame_feat.unsqueeze(0)   # [1, 1, emb_dim]
                )
                gate = self.gate_layer(ca_fused)
                visual_feat = gate * ca_fused + (1 - gate) * speaker_emb.unsqueeze(0)
                visual_feat = visual_feat.squeeze(0)  # [1, emb_dim]
            else:
                visual_feat = frame_feat  # [1, emb_dim]
            visual_features.append(visual_feat)
        
        # 按视频分组处理
        batch_features = []
        batch_logits = []
        start_idx = 0
        
        for num_frames in frame_counts:
            if num_frames == 0:
                batch_features.append(torch.zeros(self.emb_dim).to(self.device))
                batch_logits.append(torch.zeros(self.num_labels).to(self.device))
                continue
            
            # 获取该视频的所有帧特征
            video_features = visual_features[start_idx:start_idx + num_frames]  # list of [1, emb_dim]
            video_features = torch.cat(video_features, dim=0)  # [num_frames, emb_dim]
            video_features = video_features.unsqueeze(0)      # [1, num_frames, emb_dim]
            
            transformer_out = self.transformer_encoder(video_features)  # [1, num_frames, emb_dim]
            
            if self.dropout:
                transformer_out = self.dropout(transformer_out)
            
            pooled = torch.max(transformer_out, dim=1)[0]  # [1, emb_dim]
            logits = self.fc(pooled)  # [1, num_labels]
            
            batch_features.append(pooled.squeeze(0))
            batch_logits.append(logits.squeeze(0))
            
            start_idx += num_frames
        
        features = torch.stack(batch_features, dim=0)  # [batch_size, emb_dim]
        logits = torch.stack(batch_logits, dim=0)      # [batch_size, num_labels]
        
        return features, logits
    
    def forward(self, x, split=None):
        """
        统一 forward 接口
        
        Args:
            x: str (单视频路径) 或 List[str] (批量路径)
            split: 'train' / 'dev' / 'test'
        """
        if isinstance(x, str):
            return self._process_single_video(x, split)
        elif isinstance(x, list):
            return self._process_batch(x, split)
        else:
            raise TypeError(f"Unsupported input type: {type(x)}")