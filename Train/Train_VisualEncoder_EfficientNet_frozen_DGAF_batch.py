# train_with_batch.py
import os
os.environ["PYTHONIOENCODING"] = "utf-8"

import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
import sys
from sklearn.metrics import f1_score
import random

sys.path.append('Models')
from VisualEncoder_EfficientNet_frozen_DGAF_iemocap2_batch import VisualEncoderTransRefine
import argparse
import numpy as np
import pickle
from datetime import datetime
from torch.utils.data import Dataset, DataLoader


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class EmotionDataset(Dataset):
    def __init__(self, data, base_dir, split):
        self.data = data.reset_index(drop=True)
        self.base_dir = base_dir
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dia_id = row['Dialogue_ID']
        utt_id = row['Utterance_ID']
        emotion_label = row['Emotion']
        
        video_folder_path = os.path.join(
            self.base_dir, 
            f"{str(dia_id)}/{str(dia_id)}_utt{str(utt_id)}"
        )
        
        return video_folder_path, emotion_label, dia_id, utt_id


def collate_fn(batch):
    video_paths = [item[0] for item in batch]
    emotion_labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    dia_ids = [item[2] for item in batch]
    utt_ids = [item[3] for item in batch]
    return video_paths, emotion_labels, dia_ids, utt_ids


class Trainer:
    def __init__(self, model, dataset, train_data_path, dev_data_path, test_data_path, 
                 train_splits_images_dir, dev_splits_images_dir, test_splits_images_dir, 
                 num_labels, lr, weight_decay, num_epochs, poseCT, batch_size=4, 
                 accumulation_steps=1, num_workers=4):
        
        self.model = nn.DataParallel(model)
        self.dataset = dataset
        self.train_splits_images_dir = train_splits_images_dir
        self.dev_splits_images_dir = dev_splits_images_dir
        self.test_splits_images_dir = test_splits_images_dir
        
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.poseCT = poseCT
        self.batch_size = batch_size
        self.accumulation_steps = accumulation_steps
        self.num_workers = num_workers
        
        self.features_save_dir = f"Features/{dataset}_visual_features"
        self.model_save_dir = f"Best_models/{dataset}_best_visual_models"
        
        # Load data
        self.train_data = pd.read_csv(train_data_path)
        self.dev_data = pd.read_csv(dev_data_path)
        self.test_data = pd.read_csv(test_data_path)
        
        # 确保 Emotion 列是整数
        if self.train_data['Emotion'].dtype == 'object':
            label_mapping = {'hap': 0, 'sad': 1, 'neu': 2, 'ang': 3, 'exc': 4, 'fru': 5}
            self.train_data['Emotion'] = self.train_data['Emotion'].map(label_mapping)
            self.dev_data['Emotion'] = self.dev_data['Emotion'].map(label_mapping)
            self.test_data['Emotion'] = self.test_data['Emotion'].map(label_mapping)
        
        # 创建 Dataset 和 DataLoader
        train_dataset = EmotionDataset(self.train_data, self.train_splits_images_dir, 'train')
        dev_dataset = EmotionDataset(self.dev_data, self.dev_splits_images_dir, 'dev')
        test_dataset = EmotionDataset(self.test_data, self.test_splits_images_dir, 'test')
        
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True, drop_last=False
        )
        self.dev_loader = DataLoader(
            dev_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=collate_fn, pin_memory=True
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=3)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.best_f1 = 0.0
        self.best_epoch = -1
        self.best_model = None
        self.best_model_id = 0
        self.best_model_save_path = ""
        
        print(f"Device: {self.device}, Batch size: {self.batch_size}, Num workers: {self.num_workers}")
        print(f"Train: {len(self.train_data)}, Dev: {len(self.dev_data)}, Test: {len(self.test_data)}")
    
    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            print(f"\n{'='*60}\nEpoch {epoch + 1}/{self.num_epochs}\n{'='*60}")
            
            epoch_start_time = time.time()
            running_loss = 0.0
            all_labels = []
            all_preds = []
            self.optimizer.zero_grad()
            
            pbar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}", ascii=True)
            for batch_idx, (video_paths, emotion_labels, _, _) in enumerate(pbar):
                emotion_labels = emotion_labels.to(self.device)
                
                # 直接调用模型，传入视频路径列表
                _, logits = self.model(video_paths, 'train')
                
                loss = self.criterion(logits, emotion_labels)
                loss = loss / self.accumulation_steps
                loss.backward()
                
                running_loss += loss.item() * self.accumulation_steps
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(emotion_labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                current_loss = running_loss / (batch_idx + 1)
                current_f1 = f1_score(all_labels, all_preds, average='weighted') if all_labels else 0
                pbar.set_postfix({'loss': f'{current_loss:.4f}', 'f1': f'{current_f1:.4f}'})
            
            if (batch_idx + 1) % self.accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            train_f1 = f1_score(all_labels, all_preds, average='weighted')
            avg_loss = running_loss / len(self.train_loader)
            print(f"\nTrain - Loss: {avg_loss:.4f}, Weighted F1: {train_f1:.4f}")
            
            dev_f1 = self.evaluate('dev')
            print(f"Dev - Weighted F1: {dev_f1:.4f}")
            
            self.scheduler.step(dev_f1)
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if dev_f1 > self.best_f1:
                self.best_f1 = dev_f1
                self.best_epoch = epoch + 1
                self.best_model = self.model.module.state_dict().copy()
                print(f"* Best model updated! Dev F1: {self.best_f1:.4f}")
            
            print(f"Epoch took: {time.time() - epoch_start_time:.2f} seconds")
        
        self._save_best_model()
        print("\nTraining Finished!")
    
    def evaluate(self, split='dev'):
        self.model.eval()
        all_labels, all_preds = [], []
        loader = self.dev_loader if split == 'dev' else self.test_loader
        
        with torch.no_grad():
            for video_paths, emotion_labels, _, _ in tqdm(loader, desc=f"Evaluating {split}", ascii=True):
                emotion_labels = emotion_labels.to(self.device)
                _, logits = self.model(video_paths, split)
                preds = torch.argmax(logits, dim=1)
                all_labels.extend(emotion_labels.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
        
        return f1_score(all_labels, all_preds, average='weighted')
    
    def _save_best_model(self):
        self.best_model_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(self.model_save_dir, exist_ok=True)
        self.best_model_save_path = os.path.join(
            self.model_save_dir,
            f"{self.best_model_id}_best_epoch_{self.best_epoch}_with_dev_weighted_F1_{self.best_f1:.4f}_bs{self.batch_size}.pth"
        )
        torch.save(self.best_model, self.best_model_save_path)
        print(f"Best model saved to: {self.best_model_save_path}")
    
    def test(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model.module.load_state_dict(torch.load(model_path))
            print(f"Loaded model from: {model_path}")
        elif self.best_model is not None:
            self.model.module.load_state_dict(self.best_model)
            print(f"Loaded best model (epoch {self.best_epoch})")
        else:
            print("No model found!")
            return None
        
        test_f1 = self.evaluate('test')
        print(f"\n{'='*60}\nTest Weighted F1: {test_f1:.4f}\n{'='*60}")
        return test_f1
    
    def save_features(self, split='train', model_path=None, features_dict=None):
        if model_path is None or not os.path.exists(model_path):
            print("Model path does not exist.")
            return features_dict
        if features_dict is None:
            features_dict = {}
        
        self.model.module.load_state_dict(torch.load(model_path))
        print(f"Loaded {model_path} for saving {split} features.")
        self.model.eval()
        
        data = getattr(self, f'{split}_data')
        base_dir = getattr(self, f'{split}_splits_images_dir')
        ids = data['Dialogue_ID'].unique()
        
        total_features = 0  # 添加：总特征数量计数器

        # 记录特征提取开始时间
        program_start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("="*60)
        print(f"start_time: {start_datetime}")
        print("="*60)
        
        with torch.no_grad():
            for dia_id in tqdm(ids, desc=f"Saving {split} features", ascii=True):
                features_dict[dia_id] = []
                utterances = data[data['Dialogue_ID'] == dia_id]
                video_paths = [
                    os.path.join(base_dir, f"{str(dia_id)}/{str(dia_id)}_utt{str(row['Utterance_ID'])}")
                    for _, row in utterances.iterrows()
                ]
                features_list, _ = self.model(video_paths, split)
                if features_list is not None:
                    features_dict[dia_id] = np.concatenate([f.cpu().numpy() for f in features_list], axis=0)
                    total_features += len(features_list)  # 添加：累加特征数量
        
        # 结束时间
        program_end_time = time.time()
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_elapsed = program_end_time - program_start_time
        
        print("\n" + "="*60)
        print("="*60)
        print(f"end_time: {end_datetime}")
        print(f"total_time: {total_elapsed:.2f} s ({total_elapsed/60:.2f} min)")
        print("="*60)
        
        print(f"Saved features for {len(features_dict)} dialogues.")
        print(f"Total number of features saved: {total_features}.")  # 添加：打印总特征数量
        return features_dict


if __name__ == "__main__":
    program_start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("="*60)
    print(f"Start time: {start_datetime}")
    print("="*60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='IEMOCAP')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--tbb', type=float, default=0.85)
    parser.add_argument('--poseCT', type=float, default=0.85)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--test_mode', action='store_true')
    parser.add_argument('--save_features', action='store_true')
    parser.add_argument('--seed', type=int, default=888)
    parser.add_argument('--model_path', type=str, default=None)
    
    args = parser.parse_args()
    print(f"args={args}")
    
    seed_everything(args.seed)
    
    train_path = f"E:/Data/IEMOCAP.DProcess/train_emotions.csv"
    dev_path = f"E:/Data/IEMOCAP.DProcess/dev_emotions.csv"
    test_path = f"E:/Data/IEMOCAP.DProcess/test_emotions.csv"
    
    train_splits_images_dir = f"E:/Data/IEMOCAP.DProcess/train"
    dev_splits_images_dir = f"E:/Data/IEMOCAP.DProcess/dev"
    test_splits_images_dir = f"E:/Data/IEMOCAP.DProcess/test"
    
    model = VisualEncoderTransRefine(
        emb_dim=args.emb_dim, num_labels=8,
        tbb=args.tbb, poseCT=args.poseCT, dropout_rate=args.dropout_rate
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    trainer = Trainer(
        model=model, dataset=args.dataset,
        train_data_path=train_path, dev_data_path=dev_path, test_data_path=test_path,
        train_splits_images_dir=train_splits_images_dir,
        dev_splits_images_dir=dev_splits_images_dir,
        test_splits_images_dir=test_splits_images_dir,
        num_labels=8, lr=args.lr, weight_decay=args.weight_decay,
        num_epochs=args.num_epochs, poseCT=args.poseCT,
        batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
        num_workers=args.num_workers
    )
    
    if not args.test_mode:
        trainer.train()
        trainer.test()
    else:
        model_path = "E:/Code/SEE-CAT/Best_models/iemocap_best_visual_models/20260518_215419_best_epoch_30_with_dev_weighted_F1_0.1989_bs32.pth"
        trainer.test(model_path=model_path)
        
        if args.save_features:
            save_features_dict = {}
            for split in ['train', 'dev', 'test']:
                save_features_dict = trainer.save_features(split=split, model_path=model_path, features_dict=save_features_dict)
            features_save_dir = f"Features/{args.dataset}_visual_features"
            os.makedirs(features_save_dir, exist_ok=True)
            path = os.path.join(features_save_dir, f"{os.path.basename(model_path)[:15]}_model_visual_features_dict_bs{args.batch_size}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(save_features_dict, f)
            print(f"Features saved to: {path}")
    
    total_elapsed = time.time() - program_start_time
    print(f"\nTotal time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")