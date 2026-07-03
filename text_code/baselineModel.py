import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sentence_transformers import SentenceTransformer
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer
from transformers import BertModel, RobertaModel, XLNetModel
import math
import copy
import sys

if torch.cuda.is_available():
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor

else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')      ## nn.NLLLoss
        '''
        nn.NLLLoss
        官方文档中介绍称： nn.NLLLoss输入是一个对数概率向量和一个目标标签，它与nn.CrossEntropyLoss的关系可以描述为：softmax(x)+log(x)+nn.NLLLoss====>nn.CrossEntropyLoss
        '''

    def forward(self, pred, target, mask):
        '''
        param pred: (batch_size, num_utterances, n_classes)
        param target: (batch_size, num_utterances)
        param mask: (batch_size, num_utterances)
        '''
        mask_ = mask.view(-1,1) 
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)/torch.sum(self.weight[target]*mask_.squeeze())
        return loss
    
    
# 部分参考github项目 https://github.com/declare-lab/conv-emotion
class EncoderModel(nn.Module):
    def __init__(self, transformer_model_family='roberta', mode=1, attention=False, residual=False):
        '''
        param transformer_model_family: bert or roberta
        param mode: 0(base) or 1(large)
        '''
        super().__init__()
        
        if transformer_model_family == 'bert':
            if mode == '0':
                model = BertModel.from_pretrained('bert-base-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                hidden_dim = 768
            elif mode == '1':
                model = BertModel.from_pretrained('bert-large-uncased')
                tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
                hidden_dim = 1024       
        elif transformer_model_family == 'roberta':
            if mode == '0':
                model = RobertaModel.from_pretrained('roberta-base')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
                hidden_dim = 768
            elif mode == '1':
                model = RobertaModel.from_pretrained('roberta-large')
                tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
                hidden_dim = 1024      
                
        self.transformer_model_family = transformer_model_family
        self.model = model.cuda()
        self.hidden_dim = hidden_dim
        self.residual = residual
        
        if self.transformer_model_family in ['bert', 'roberta']:
            self.tokenizer = tokenizer
        
    def pad(self, tensor, length):
        if length > tensor.size(0):
            return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
        else:
            return tensor
        
    def forward(self, conversations, lengths):
        '''
        param conversations: 包含语句序列的对话，list
        param lengths: 每个对话的长度，list
        返回值：经过PLM编码后的语义向量，及掩码mask
        '''
        # 提取多个对话中的所有句子
        lengths = torch.Tensor(lengths).long()
        start = torch.cumsum(torch.cat((lengths.data.new(1).zero_(), lengths[:-1])), 0)
        utterances = [sent for conv in conversations for sent in conv]\
        
        if self.transformer_model_family in ['bert', 'roberta']:
            # 分词
            batch = self.tokenizer(utterances, max_length=512, truncation=True, padding=True, return_tensors="pt") 
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            # 返回[CLS]位的向量作为语义向量
            # self.model: RobertaForSequenceClassification
            # _, features = self.model(input_ids, attention_mask, output_hidden_states=True) 
            # if self.transformer_model_family == 'roberta':
            #     features = features[:, 0, :]
            # 替换成PLM原始模型（BertModel; RobertaModel）后重新写这部分
            # transformers 4.24.0
            outputs = self.model(input_ids, attention_mask, output_hidden_states=True) 
            sequence_output = outputs[0]  # batch, seq_len, dim
            cls = sequence_output[:, 0, :]  # batch, dim  [取最后一层隐藏层的第一位，即cls对应的位置]
            
        # print(cls.shape)
        # torch.Size([4, 1024])

        # 把输出的features重新组织成batch的形式，(total_utterances_num, hidden_dim) -> (utterances_num_per_conversation, batch_size, hidden_dim)
        features = torch.stack([self.pad(cls.narrow(0, s, l), max(lengths))
                                for s, l in zip(start.data.tolist(), lengths.data.tolist())], 0).transpose(0, 1)
    
        # print(features.shape)
        # torch.Size([1, 4, 1024])

        return cls, features

class LinearClassifer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.smax_fc = nn.Linear(input_dim, num_classes).cuda()

    def forward(self, input):
        logits = self.smax_fc(input) # 映射至标签空间
        log_prob = F.log_softmax(logits, 2) # 归一化得到概率向量
        return log_prob, logits # 返回概率向量，和logits

class baselineModel(nn.Module):
    def __init__(self, transformer_model_family, mode, num_classes, attention=False, residual=False):

        '''
        param D_h: lstm隐藏层大小
        param transformer_model_family: bert or roberta or xlnet
        param mode: 0(base) or 1(large)
        param num_classes: num of emotion classes 
        param num_subclasses: num of emotion bias classes
        param context_encoder_layer: context transformer layer size
        '''
        super().__init__()
        # 语句编码器
        self.encoderModel = EncoderModel(transformer_model_family, mode, attention, residual)
        
        # 情感偏移感知任务
        if mode == '0':
            hidden_dim = 768
        elif mode == '1':
            hidden_dim = 1024
        
        self.mainClassifer = LinearClassifer(hidden_dim, num_classes)

    def forward(self, conversations, lengths):
        '''
        param conversations: 包含语句序列的对话，list   ->  转换成4句一组
        param lengths: 每个对话的长度，list
        param subindex: 每个句子同一说话者上句的编号，tensor, (utterances_num_per_conversation, batch_size)
        返回值：两个任务的概率向量
        '''
        # 语句编码器
        cls, features = self.encoderModel(conversations, lengths) # 输入对话中的语句，返回bert编码后的语义向量
        # print(features.shape)
        # max_utt_num x batch_size x hidden_size
    
        main_output, _ = self.mainClassifer(features) # 分类器，返回的main_output是情感类别概率
        # print(main_output.shape)
        # max_utt_num x batch_size x n_classes

        return cls, main_output
