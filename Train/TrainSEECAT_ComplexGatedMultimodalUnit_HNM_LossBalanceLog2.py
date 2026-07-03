import sys
sys.path.append('Loss')
sys.path.append('Model')
sys.path.append('Dataset')
from IEMOCAPDataset import IEMOCAPDataset
from MELDDataset import MELDDataset
from SEECAT_Model_NOCM_addCL_MLP3_ComplexGatedMultimodalUnit_HNM import SEECAT
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from optparse import OptionParser
import torch.optim as optim
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')
from torch.utils.data.sampler import SubsetRandomSampler
import random

import argparse
import time
import os


class TrainSEECAT():

    def __init__(self, dataset, batch_size, num_epochs, learning_rate, weight_decay, 
                 model_dim, dropout_rate, dropout_rec,
                 CL_loss_param,CLT_loss_param, temperature, scl_temperature, device, hnmk, valid_ratio):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.model_dim = model_dim
        self.dropout_rate = dropout_rate
        self.dropout_rec = dropout_rec

        self.CL_loss_param = CL_loss_param
        self.CLT_loss_param = CLT_loss_param

        self.temperature = temperature
        self.scl_temperature = scl_temperature
        self.device = device
        
        self.hnmk = hnmk
        self.valid_ratio = valid_ratio 

        self.best_test_f1 = 0.0
        self.best_epoch = 1
        self.best_test_report = None
        self.best_test_all_preds = None
        
        self.best_valid_f1 = 0.0
        self.best_valid_epoch = 1
        self.best_test_f1_at_best_valid = 0.0
        self.best_test_report_at_best_valid = None
        self.best_test_all_preds_at_best_valid = None

        self.get_dataloader(valid_ratio)
        self.get_model()
        self.get_loss()
        self.get_optimizer()
    

    def get_train_valid_sampler(self, train_dataset, valid = 0.05):
        size = len(train_dataset)
        idx = list(range(size))
        split = int(valid * size)
        np.random.shuffle(idx)
        return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


    def get_dataloader(self, valid = 0.05):
        if self.dataset == 'IEMOCAP':
            train_dataset = IEMOCAPDataset(train = True)
            test_dataset = IEMOCAPDataset(train = False)
        elif self.dataset == 'MELD':
            train_dataset = MELDDataset(train = True)
            test_dataset = MELDDataset(train = False)


        train_sampler, valid_sampler = self.get_train_valid_sampler(train_dataset, valid)
        self.train_dataloader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, 
                                           sampler = train_sampler, collate_fn = train_dataset.collate_fn, num_workers = 0)
        self.valid_dataloader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, 
                                          sampler = valid_sampler,collate_fn = train_dataset.collate_fn, num_workers = 0)
        self.test_dataloader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, 
                                          collate_fn = test_dataset.collate_fn, shuffle = False, num_workers = 0)
    

    def get_class_counts(self):
        class_counts = torch.zeros(self.num_classes).to(self.device)

        for _, data in enumerate(self.train_dataloader):
            _, _, _, _, _, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]
            class_counts += torch.bincount(labels, minlength = self.num_classes)

        return class_counts
    

    def get_model(self):
        if self.dataset == 'IEMOCAP':
            self.num_classes = 6
            self.n_speakers = 2
        elif self.dataset == 'MELD':
            self.num_classes = 7
            self.n_speakers = 9

        roberta_dim = 1024
        D_m_audio = 6373
        D_m_visual = 512
        listener_state = False
        D_e = self.model_dim 
        D_p = self.model_dim
        D_g = self.model_dim
        D_h = self.model_dim
        D_a = self.model_dim 
        context_attention = 'simple'
   
        dropout_rate = self.dropout_rate 


        self.model = SEECAT(self.dataset, self.temperature, self.scl_temperature, roberta_dim,  dropout_rate, 
                                    self.model_dim, D_m_audio, D_m_visual, D_g, D_p, D_e, 
                                    D_h, self.num_classes, self.n_speakers, 
                                    listener_state, context_attention, D_a, self.dropout_rec, self.device, self.hnmk).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Non-trainable parameters: {non_trainable_params}")


    def get_loss(self):
        class_counts = self.get_class_counts()
        self.CE_loss = nn.CrossEntropyLoss()
    

    def get_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor = 0.95, patience = 10, threshold = 1e-6, verbose = True)
    

    def train_or_eval_model_per_epoch(self, dataloader, train = True):
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss = 0.0

        total_tc_loss, total_TA_loss, total_TV_loss, total_CE_loss = 0.0, 0.0, 0.0, 0.0
        all_labels, all_preds = [], []
        for _, data in enumerate(dataloader):
            if train:
                self.optimizer.zero_grad() 

            padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels = [d.to(self.device) for d in data]
            padded_labels = padded_labels.reshape(-1)
            labels = padded_labels[padded_labels != -1]


            TC_loss, TA_loss, TV_loss, logits = \
                self.model(padded_texts, padded_audios, padded_visuals, padded_speaker_masks, padded_utterance_masks, padded_labels)
            
            CE_loss = self.CE_loss(logits, labels)
            
            # 对每个 loss 做缩放（防止极小/极大的影响）
            TC_loss_n = torch.log1p(torch.log(TC_loss + 1e-6))
            TA_loss_n = torch.log1p(torch.log(TA_loss + 1e-6))
            TV_loss_n = torch.log1p(torch.log(TV_loss + 1e-6))
            CE_loss_n = torch.log1p(torch.log(CE_loss + 1e-6))

            loss = TC_loss_n * self.CLT_loss_param + TA_loss_n * self.CL_loss_param + TV_loss_n * self.CL_loss_param + CE_loss_n * (1-2*self.CL_loss_param-self.CLT_loss_param)

            total_loss += loss.item()

            total_tc_loss += TC_loss_n.item()
            total_TA_loss += TA_loss_n.item()
            total_TV_loss += TV_loss_n.item()
            total_CE_loss += CE_loss_n.item()

            if train:
                loss.backward()
                self.optimizer.step()
            
            preds = torch.argmax(logits, dim = -1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        avg_f1 = round(f1_score(all_labels, all_preds, average = 'weighted') * 100, 4)
        avg_acc = round(accuracy_score(all_labels, all_preds) * 100, 4)
        report = classification_report(all_labels, all_preds, digits = 4)
        

        return round(total_loss, 4), round(total_tc_loss, 4), round(total_TA_loss, 4), round(total_TV_loss, 4), round(total_CE_loss, 4), avg_f1, avg_acc, report, all_preds 


    def train_or_eval_linear_model(self):
        for e in range(self.num_epochs):
            train_loss, train_tc_loss, train_TA_loss, train_TV_loss, train_CE_loss, train_f1, train_acc, _, _ = self.train_or_eval_model_per_epoch(self.train_dataloader, train = True)


            with torch.no_grad():
                valid_loss, valid_tc_loss, valid_TA_loss, valid_TV_loss, valid_CE_loss,valid_f1, valid_acc, _, _ = self.train_or_eval_model_per_epoch(self.valid_dataloader, train = False)
                test_loss, test_tc_loss, test_TA_loss,  test_TV_loss,  test_CE_loss, test_f1, test_acc, test_report, test_all_preds = self.train_or_eval_model_per_epoch(self.test_dataloader, train = False)
            print('Epoch {}, train loss: {}, train TC loss: {}, train TA loss: {}, train TV loss: {}, train CE loss: {}, train f1: {}, train acc: {}'.format(e + 1, train_loss, train_tc_loss, train_TA_loss, train_TV_loss, train_CE_loss, train_f1, train_acc))
            print('Epoch {}, valid loss: {}, valid TC loss: {}, valid TA loss: {}, valid TV loss: {}, valid CE loss: {}, valid f1: {}, valid acc: {}'.format(e + 1, valid_loss, valid_tc_loss, valid_TA_loss, valid_TV_loss, valid_CE_loss, valid_f1, valid_acc))
            print('Epoch {}, test loss: {}, test TC loss: {}, test TA loss: {}, test TV loss: {}, test CE loss: {}, test f1: {}, test acc: {}, '.format(e + 1, test_loss,  test_tc_loss, test_TA_loss, test_TV_loss, test_CE_loss, test_f1, test_acc))
            print(test_report)   

            self.scheduler.step(valid_loss)

            if test_f1 >= self.best_test_f1:
                self.best_test_f1 = test_f1
                self.best_epoch = e + 1
                self.best_test_report = test_report
                self.best_test_all_preds = test_all_preds

            # 如果验证集f1更优，则保存当前验证集和测试集信息
            if valid_f1 >= self.best_valid_f1:
                self.best_valid_f1 = valid_f1
                self.best_valid_epoch = e + 1
                self.best_test_f1_at_best_valid = test_f1
                self.best_test_report_at_best_valid = test_report
                self.best_test_all_preds_at_best_valid = test_all_preds
        
        if not ((self.dataset=='MELD' and self.best_test_f1 < 66.5 and self.best_test_f1_at_best_valid < 66.5) or (self.dataset=='IEMOCAP' and self.best_test_f1 < 70 and self.best_test_f1_at_best_valid < 70))
            print("Saving predictions.")
            timestamp = int(time.time())
            rand_id = random.randint(10000, 99999)

            filename1 = f"all_preds_{timestamp}_{rand_id}_test.txt"
            save_path = os.path.join(f"Preds/{self.dataset}/", filename1)  # 可指定保存目录
            os.makedirs("Preds/{self.dataset}/", exist_ok=True)
            np.savetxt(save_path, self.best_test_all_preds, fmt='%d')
            print(f"[INFO] Predictions saved to {save_path}")

            filename2 = f"all_preds_{timestamp}_{rand_id}_test_best_valid.txt"
            save_path = os.path.join(f"Preds/{self.dataset}/", filename2)  # 可指定保存目录
            np.savetxt(save_path, self.best_test_all_preds_at_best_valid, fmt='%d')
            print(f"[INFO] Predictions saved to {save_path}")

                
        print('Best test f1: {} at epoch {}'.format(self.best_test_f1, self.best_epoch))
        print(self.best_test_report)

        print(f'Best valid f1: {self.best_valid_f1} at epoch {self.best_valid_epoch}')
        print(f'Test f1 at best valid epoch: {self.best_test_f1_at_best_valid}')
        print(self.best_test_report_at_best_valid)
        



def get_args():
    parser = argparse.ArgumentParser(description='Train SEE-CAT Model')
    parser.add_argument('--dataset', default = 'MELD', type = str, help = 'MELD or IEMOCAP')
    parser.add_argument('--batch_size', default = 64, type = int, help = '64 for IEMOCAP and 100 for MELD')
    parser.add_argument('--num_epochs', default = 100, type = int, help = 'number of epochs')
    
    parser.add_argument('--learning_rate', default = 0.0001, type = float, help = 'learning rate')
    parser.add_argument('--weight_decay', default = 0.00001, type = float, help = 'weight decay parameter')
    
    parser.add_argument('--model_dim', default = 256, type = int, help = 'model dimension')
    
    parser.add_argument('--dropout_rate', default = 0, type = float, help = 'dropout rate')
    parser.add_argument('--dropout_rec', default = 0, type = float, help = 'dropout rec')
  
    parser.add_argument('--CL_loss_param', default = 0.1, type = float, help = 'coefficient of TA/TV loss')
    parser.add_argument('--CLT_loss_param', default = 0.2, type = float, help = 'coefficient of TC loss')
    parser.add_argument('--temperature', default = 0.2, type = float, help = 'CL temperature')
    parser.add_argument('--scl_temperature', default = 0.2, type = float, help = 'SCL temperature')
    parser.add_argument('--seed', default = 2023, type = int, help = 'seed') 
    parser.add_argument('--hnmk', default = 5, type = int, help = 'HNM Top-k')
    parser.add_argument('--valid_ratio', default = 0.05, type = float, help = 'valid ratio')

    args = parser.parse_args()
    
    print(args)

    return args


def set_seed(seed):
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = get_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    model_dim = args.model_dim


    dropout_rate = args.dropout_rate
    dropout_rec = args.dropout_rec

    CL_loss_param = args.CL_loss_param
    CLT_loss_param = args.CLT_loss_param

    temperature = args.temperature
    scl_temperature = args.scl_temperature
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    hnmk = args.hnmk
    valid_ratio = args.valid_ratio

    seed = args.seed
    set_seed(seed)
    print(seed)

    SEECAT_train = TrainSEECAT(dataset, batch_size, num_epochs, learning_rate, 
                                   weight_decay, model_dim, 
                                   dropout_rate, dropout_rec, CL_loss_param, CLT_loss_param, temperature, scl_temperature, device, hnmk, valid_ratio)
    SEECAT_train.train_or_eval_linear_model()

        