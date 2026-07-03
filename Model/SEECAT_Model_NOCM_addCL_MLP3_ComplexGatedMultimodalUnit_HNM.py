from DialogueRNN import BiModel
from MLP3 import MLP

import torch
import torch.nn as nn
import sys
import torch.nn.functional as F


def contrastive_loss(anchor, positive, temperature=6):
    """
    Calculate contrastive loss between anchor (e.g., text) and positive (e.g., audio or visual) features.
    Based on the formula for contrastive loss with cosine similarity:
    
    L = - (1/N) * sum(log(exp(sim(h_i^t, h_i^a) / tau) / sum_j exp(sim(h_i^t, h_j^a) / tau)))
    
    Args:
    - anchor: Tensor of shape [batch_size, seq_len, feature_dim] (e.g., text features).
    - positive: Tensor of shape [batch_size, seq_len, feature_dim] (e.g., audio or visual features).
    - temperature: A scalar for scaling the similarities (tau).
    
    Returns:
    - loss: Contrastive loss for the batch.
    """
    
    # Normalize anchor and positive feature vectors
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    
    # Calculate cosine similarity for each i and j
    # This creates a similarity matrix of shape [batch_size, seq_len, seq_len] for every timestep pair
    similarity_matrix = torch.matmul(anchor, positive.transpose(-1, -2)) / temperature  # Cosine similarity / tau

    # Labels are diagonal (matching i-th text with i-th audio/visual)
    batch_size, seq_len = similarity_matrix.size(0), similarity_matrix.size(1)
    labels = torch.arange(seq_len).expand(batch_size, -1).to(similarity_matrix.device)  # Shape: [batch_size, seq_len]

    # Cross entropy loss
    loss = F.cross_entropy(similarity_matrix.reshape(-1, seq_len), labels.reshape(-1))
    
    return loss

def supervised_contrastive_loss_with_hard_neg(features, labels, temperature=6, top_k=5):
    """
    Supervised Contrastive Loss with Hard Negative Mining
    - features: [N, D]
    - labels: [N]
    """
    device = features.device
    features = F.normalize(features, dim=-1)
    sim_matrix = torch.matmul(features, features.T) / temperature  # [N, N]

    labels = labels.view(-1, 1)  # [N, 1]
    mask_pos = torch.eq(labels, labels.T).float()  # [N, N]，正样本掩码
    mask_neg = 1.0 - mask_pos  # 负样本掩码

    logits = sim_matrix

    # Mask对角线（自己与自己）不参与
    logits_mask = torch.ones_like(mask_neg) - torch.eye(len(features), device=device)
    mask_pos *= logits_mask
    mask_neg *= logits_mask

    # ----------------------------
    # Hard Negative Mining 部分：
    # ----------------------------
    # 对于每一行 anchor，从负样本中选出 top-k 最难的（最相似的）
    hard_neg_logits = logits.masked_fill(mask_pos.bool(), float('-inf'))  # 只保留负样本
    topk_neg_values, _ = torch.topk(hard_neg_logits, min(top_k, logits.shape[1]-1), dim=1)

    # Positive 对比
    exp_logits = torch.exp(logits)
    log_prob = logits - torch.log(
        torch.exp(topk_neg_values).sum(dim=-1, keepdim=True) + (exp_logits * mask_pos).sum(dim=-1, keepdim=True)
    )

    # 只用正样本计算 loss
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / (mask_pos.sum(dim=1) + 1e-8)

    loss = -mean_log_prob_pos.mean()
    return loss




class GMU3Modal(nn.Module):
    def __init__(self, input_dim, model_dim, dropout=0.1):
        super(GMU3Modal, self).__init__()
        
        self.input_dim = input_dim
        self.model_dim = model_dim

        # 模态独立变换
        self.t_proj = nn.Linear(input_dim, model_dim)
        self.a_proj = nn.Linear(input_dim, model_dim)
        self.v_proj = nn.Linear(input_dim, model_dim)

        # 融合门控
        self.z_gate = nn.Linear(3 * model_dim, 3)  # 控制模态加权
        self.fusion_proj = nn.Linear(3 * model_dim, model_dim)  # 对concat后表示线性变换

        # 残差调节门
        self.r_gate = nn.Linear(model_dim * 2, model_dim)

        # 激活、Dropout、LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, t, a, v):
        # 投影每个模态到统一 model_dim 空间
        t_ = F.relu(self.t_proj(t))
        a_ = F.relu(self.a_proj(a))
        v_ = F.relu(self.v_proj(v))

        # 拼接三模态向量计算门控
        concat = torch.cat([t_, a_, v_], dim=-1)  # [B, 3*D]
        gates = torch.softmax(self.z_gate(concat), dim=-1)  # [B, 3]

        # 门控加权求融合表示
        fused = gates[:, 0].unsqueeze(-1) * t_ + gates[:, 1].unsqueeze(-1) * a_ + gates[:, 2].unsqueeze(-1) * v_

        # 对 concat 做线性变换，加入非线性再 residual 融合
        fusion_enhanced = F.gelu(self.fusion_proj(concat))
        fusion_enhanced = self.dropout(fusion_enhanced)

        # 残差门控
        res_input = torch.cat([fused, fusion_enhanced], dim=-1)
        residual_gate = torch.sigmoid(self.r_gate(res_input))

        # 最终融合：残差 + 门控
        output = residual_gate * fusion_enhanced + (1 - residual_gate) * fused

        # LayerNorm
        output = self.layer_norm(output)
        return output

class SEECAT(nn.Module):

    def __init__(self, dataset, temperature, scl_temperature, roberta_dim, dropout,
                 model_dim, D_m_audio, D_m_visual, D_g, D_p, D_e, D_h,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec, device, k=5):
        super().__init__()

        self.dataset = dataset  
        self.temperature = temperature
        self.scl_temperature = scl_temperature
        self.k = k

        self.text_fc = nn.Linear(roberta_dim, model_dim)
        self.audio_fc = nn.Linear(D_m_audio, model_dim)
        self.audio_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.visual_fc = nn.Linear(D_m_visual, model_dim)
        self.visual_dialoguernn = BiModel(model_dim, D_g, D_p, D_e, D_h, dataset,
                 n_classes, n_speakers, listener_state, context_attention, D_a, dropout_rec,
                 dropout, device)
        
        self.gmu = GMU3Modal(model_dim, model_dim)

        self.mlp = MLP(model_dim, model_dim, n_classes, dropout)

        self.softmax = nn.Softmax(dim=-1)

 
    def forward(self, texts, audios, visuals, speaker_masks, utterance_masks, padded_labels):   
       
        text_features = self.text_fc(texts)
        audio_features = self.audio_fc(audios)
        visual_features = self.visual_fc(visuals)

        audio_features = self.audio_dialoguernn(audio_features, speaker_masks, utterance_masks)
        visual_features = self.visual_dialoguernn(visual_features, speaker_masks, utterance_masks)

        text_features = text_features.transpose(0, 1)
        audio_features = audio_features.transpose(0, 1)
        visual_features = visual_features.transpose(0, 1)


        text_audio_loss = contrastive_loss(text_features, audio_features, self.temperature)
        text_visual_loss = contrastive_loss(text_features, visual_features, self.temperature)

        text_features = text_features.reshape(-1, text_features.shape[-1])
        text_features = text_features[padded_labels != -1]

        text_contrastive_loss = supervised_contrastive_loss_with_hard_neg(text_features, padded_labels[padded_labels != -1], self.scl_temperature, top_k=self.k)

        audio_features= audio_features.reshape(-1, audio_features.shape[-1])
        audio_features= audio_features[padded_labels != -1]
        visual_features = visual_features.reshape(-1, visual_features.shape[-1])
        visual_features= visual_features[padded_labels != -1]

        fused_features = self.gmu(text_features, audio_features, visual_features)

        mlp_outputs = self.mlp(fused_features)
        logits = self.softmax(mlp_outputs)

        return  text_contrastive_loss, text_audio_loss, text_visual_loss, logits