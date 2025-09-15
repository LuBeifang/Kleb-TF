import torch
import torch.nn as nn
# from conformer import Conformer

# class AttachedModel(torch.nn.Module):
#     def __init__(self):
#         super(AttachedModel, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv1d(1, 32, kernel_size=16, padding=1),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.MaxPool1d(2),
#             nn.AdaptiveAvgPool1d(16),
#             nn.Flatten(),
#         )


#         self.fc = torch.nn.Sequential(
#             nn.Linear(512, 256),  # 包含交互特征
#             nn.BatchNorm1d(256),  # 添加批归一化
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, 32),
#             nn.BatchNorm1d(32),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(32, 1)
#         )

#     def forward(self, tf_embeddings, peak_embeddings):
#         # conformer out put (batch, length, dim)
#         tf_feat = self.conv(tf_embeddings.unsqueeze(1)) #  (batch, 512)
#         peak_feat = self.conv(peak_embeddings.unsqueeze(1))  #  (batch, 512)


#         hadamard = (tf_feat * peak_feat).flatten(1)  # (16,512)


#         # only hadamard
#         x = self.fc(hadamard) # (16,512)

#         x = torch.sigmoid(x)
#         return x

# class AttachedModel(nn.Module):
#     def __init__(self, embed_dim=512, d_model=256, n_heads=4, dropout=0.5, 
#                  cross_reduce_len=64):
#         """
#         embed_dim: 输入嵌入的维度（你的嵌入维度：512）
#         d_model: 融合后的特征维度
#         n_heads: 注意力头数
#         dropout: 注意力的 dropout
#         cross_reduce_len: 如不为 None，当长度 L > cross_reduce_len 时，使用 AdaptiveAvgPool1d 将长度压缩
#         """
#         super(AttachedModel, self).__init__()

#         self.embed_dim = embed_dim
#         self.d_model = d_model
#         self.cross_reduce_len = cross_reduce_len

#         self.tf_proj = nn.Linear(embed_dim, d_model)
#         self.dna_proj = nn.Linear(embed_dim, d_model)

#         self.attn_tf_to_dna = nn.MultiheadAttention(embed_dim=d_model, 
#                                                   num_heads=n_heads, 
#                                                   batch_first=True, 
#                                                   dropout=dropout)
#         self.attn_dna_to_tf = nn.MultiheadAttention(embed_dim=d_model, 
#                                                   num_heads=n_heads, 
#                                                   batch_first=True, 
#                                                   dropout=dropout)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)

#         # 融合头：将两路汇聚向量拼接后再经过全连接
#         self.fc = nn.Sequential(
#             nn.Linear(2 * d_model, 256),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(256, 1)
#         )

#         if self.cross_reduce_len is not None:
#             self.pool1d = nn.AdaptiveAvgPool1d(self.cross_reduce_len)

#     def _reduce_seq(self, x):
#         """
#         x: (B, L, D)
#         返回: (B, L', D)，若 L <= L' 则不变
#         """
#         if self.cross_reduce_len is None:
#             return x
#         B, L, D = x.size()
#         if L <= self.cross_reduce_len:
#             return x
#         # 转成 (B, D, L) 以便 AdaptiveAvgPool1d
#         x = x.transpose(1, 2)  # (B, D, L)
#         x = self.pool1d(x)       # (B, D, L')
#         x = x.transpose(1, 2)  # (B, L', D)
#         return x

#     def forward(self, tf_emb, dna_emb):
#         """
#         tf_emb: (B, Lf, D) 或 (B, D)
#         dna_emb: (B, Ld, D) 或 (B, D)
#         输出: logits (B, 1)
#         """
#         # 兼容 2D/3D 输入
#         if tf_emb.dim() == 2:
#             tf_emb = tf_emb.unsqueeze(1)  # (B, 1, D)
#         if dna_emb.dim() == 2:
#             dna_emb = dna_emb.unsqueeze(1)  # (B, 1, D)

#         # 可能的长度压缩
#         if self.cross_reduce_len is not None:
#             tf_emb = self._reduce_seq(tf_emb)
#             dna_emb = self._reduce_seq(dna_emb)

#         # 投影到统一维度
#         tf_q = self.tf_proj(tf_emb)     # (B, Lf, d_model)
#         dna_k = self.dna_proj(dna_emb)  # (B, Ld, d_model)

#         # 跨模态注意力
#         tf_attn_out, _ = self.attn_tf_to_dna(query=tf_q, key=dna_k, value=dna_k)  # (B, Lf, d_model)
#         dna_attn_out, _ = self.attn_dna_to_tf(query=dna_k, key=tf_q, value=tf_q)  # (B, Ld, d_model)

#         # 残差+归一化
#         tf_out = self.norm1(tf_attn_out + tf_q)     # (B, Lf, d_model)
#         dna_out = self.norm2(dna_attn_out + dna_k)  # (B, Ld, d_model)

#         # 全局池化
#         tf_pooled = tf_out.mean(dim=1)   # (B, d_model)
#         dna_pooled = dna_out.mean(dim=1) # (B, d_model)

#         fused = torch.cat([tf_pooled, dna_pooled], dim=-1)  # (B, 2*d_model)

#         logits = self.fc(fused)  # (B, 1)
#         return logits

class AttachedModel(nn.Module):
    """
    单向跨模态注意力：TF_emb -> DNA_emb
    输入：
      tf_emb: (B, Lf, D_in) 或 (B, D_in)
      dna_emb: (B, Ld, D_in) 或 (B, D_in)
    输出：
      logits: (B, 1)
    return_attn:
      True 时，返回 (logits, attn_weights)，attn_weights 的形状通常为 (B, N_heads, Lf, Ld)
    """
    def __init__(self, embed_dim=512, d_model=256, n_heads=4, dropout=0.3, cross_reduce_len=256, direction='tf_to_dna'):
        super(AttachedModel, self).__init__()
        self.direction = direction
        self.cross_reduce_len = cross_reduce_len

        self.tf_proj = nn.Linear(embed_dim, d_model)
        self.dna_proj = nn.Linear(embed_dim, d_model)

        # 仅实现 TF -> DNA 的单向注意力
        self.attn_tf_to_dna = nn.MultiheadAttention(embed_dim=d_model, 
                                                  num_heads=n_heads, 
                                                  batch_first=True, 
                                                  dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # 最后的小 MLP，输出一个二分类 logits
        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

        if self.cross_reduce_len is not None:
            self.pool1d = nn.AdaptiveAvgPool1d(cross_reduce_len)

    def _reduce_seq(self, x):
        """
        x: (B, L, D)
        返回: (B, L', D)，当 L > L' 时才压缩
        """
        if self.cross_reduce_len is None:
            return x
        B, L, D = x.size()
        if L <= self.cross_reduce_len:
            return x
        x = x.transpose(1, 2)  # (B, D, L)
        x = self.pool1d(x)      # (B, D, L')
        x = x.transpose(1, 2)   # (B, L', D)
        return x

    def forward(self, tf_emb, dna_emb, return_attn=False):
        """
        tf_emb: (B, Lf, D) 或 (B, D)
        dna_emb: (B, Ld, D) 或 (B, D)
        """
        # 兼容 2D/3D 输入
        if tf_emb.dim() == 2:
            tf_emb = tf_emb.unsqueeze(1)  # (B, 1, D)
        if dna_emb.dim() == 2:
            dna_emb = dna_emb.unsqueeze(1)  # (B, 1, D)

        if self.cross_reduce_len is not None:
            tf_emb = self._reduce_seq(tf_emb)
            dna_emb = self._reduce_seq(dna_emb)

        tf_q = self.tf_proj(tf_emb)   # (B, Lf, d_model)
        dna_k = self.dna_proj(dna_emb) # (B, Ld, d_model)

        # TF -> DNA 跨模态注意力（需要权重）
        attn_out, attn_weights = self.attn_tf_to_dna(query=tf_q, key=dna_k, value=dna_k, need_weights=True)

        # 残差 + 归一化
        tf_out = self.norm1(attn_out + tf_q)  # (B, Lf, d_model)

        # 池化得到向量表示
        tf_pooled = tf_out.mean(dim=1)  # (B, d_model)

        logits = self.fc(tf_pooled)    # (B, 1)

        if return_attn:
            return logits, attn_weights
        else:
            return logits