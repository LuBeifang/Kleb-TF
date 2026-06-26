import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class DNAConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.dropout(out)
        return out + residual


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer with feed-forward network."""
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        query_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if query_mask is not None:
            query = query * (~query_mask).unsqueeze(-1).float()
        attn_out, attn_w = self.attn(
            query, key_value, key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return (x, attn_w) if need_weights else (x, None)


class AttentivePoolWithScores(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1, bias=False)
        )

    def forward(self, x):
        raw_scores = self.score(x).squeeze(-1)
        weights = torch.softmax(raw_scores, dim=-1).unsqueeze(1)
        pooled = torch.bmm(weights, x).squeeze(1)
        return pooled, raw_scores, weights

class ProteinTFProjection(nn.Module):
    """Transformer-based projection head for ESM2 protein embeddings (320->512)."""
    def __init__(
        self,
        input_dim: int = 320,
        output_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_len: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, protein_emb: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(protein_emb)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        return x

class SimpleLinearProjection(nn.Module):
    """Linear projection + LayerNorm."""
    def __init__(self, input_dim=1280, output_dim=512, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        return self.dropout(self.norm(self.linear(x)))


class MLPProjection(nn.Module):
    """Multi-layer MLP projection."""
    def __init__(self, input_dim=1280, output_dim=512, hidden_dims=(1024, 768), dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.LayerNorm(hd))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev = hd
        layers.append(nn.Linear(prev, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, src_key_padding_mask=None):
        return self.mlp(x)

PROTEIN_EMBEDDING_DIMS = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
    "Rostlab/prot_bert": 1024,
    "Rostlab/prot_bert_bfd": 1024,
    "Rostlab/prot_t5_xl_uniref50": 1024,
    "ESM_DBP": 1280,
    "ESM_DBP_LORA": 1280,
}


def _build_tf_proj(proj_type, tf_input_dim, d_model, dropout):
    if proj_type == "simple_linear":
        return SimpleLinearProjection(tf_input_dim, d_model, dropout)
    elif proj_type == "transformer":
        return ProteinTFProjection(
            input_dim=tf_input_dim, output_dim=d_model,
            n_layers=2, n_heads=8, dropout=dropout,
        )
    elif proj_type == "mlp":
        return MLPProjection(tf_input_dim, d_model, dropout=dropout)
    else:
        raise ValueError(f"Unknown proj_type: {proj_type}")

class AttachedModel(nn.Module):
    """Bidirectional cross-attention with per-TF classifier heads and a shared auxiliary head."""
    def __init__(
        self,
        tf_input_dim: int = 1280,
        peak_input_dim: int = 5,
        d_model: int = 512,
        dna_encoding_method: str = "onehot",
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        protein_length: int = 903,
        dna_max_length: int = 201,
        dropout: float = 0.1,
        proj_type: str = "simple_linear",
        num_tfs: int = 10,
    ):
        super().__init__()
        if dna_encoding_method == "onehot":
            self.dna_encoding_method = 'onehot'
            self.peak_input_dim = 5
        else:
            self.dna_encoding_method = 'transformer'
            self.peak_input_dim = 512

        self.d_model = d_model
        self.protein_length = protein_length
        self.dna_max_length = dna_max_length
        self.num_tfs = num_tfs

        self.tf_proj = _build_tf_proj(proj_type, tf_input_dim, d_model, dropout)

        self.dna_conv = nn.Sequential(
            nn.Conv1d(self.peak_input_dim, 64, kernel_size=7, padding=3),
            nn.GELU(),
            nn.BatchNorm1d(64),
            DNAConvBlock(64, 128, kernel_size=5),
            nn.MaxPool1d(kernel_size=2, stride=2),
            DNAConvBlock(128, 256, kernel_size=3),
            nn.MaxPool1d(kernel_size=2, stride=2),
            DNAConvBlock(256, 384, kernel_size=3, dilation=2),
            nn.Conv1d(384, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.2),
        )
        if self.dna_encoding_method == 'onehot':
            self.dna_output_len = self._compute_conv_output_length(dna_max_length)
        else:
            self.dna_output_len = dna_max_length

        self.dna_pos_enc = PositionalEncoding(d_model, max_len=1024, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.2, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.cross_layer_tf = CrossAttentionLayer(d_model, nhead, dim_feedforward * 3, 0.1)
        self.cross_layer_dna = CrossAttentionLayer(d_model, nhead, dim_feedforward * 3, 0.1)

        self.pool = AttentivePoolWithScores(d_model)
        #TF specific head
        self.tf_specific_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 512),     
                nn.LayerNorm(512),
                nn.GELU(),                    
                nn.Dropout(0.15),
                
                nn.Linear(512, 1)
            ) for _ in range(num_tfs)
        ])
        # Shared auxiliary head
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _compute_conv_output_length(self, input_length: int) -> int:
        x = torch.zeros(1, self.peak_input_dim, input_length)
        with torch.no_grad():
            x = self.dna_conv(x)
        return x.shape[-1]

    def encode_dna(self, peak_embeddings):
        if peak_embeddings.dim() == 3 and peak_embeddings.size(-1) == 5:
            x = peak_embeddings.transpose(1, 2)
            x = self.dna_conv(x)
            x = x.transpose(1, 2)
            return x
        else:
            return peak_embeddings

    def forward(self, tf_embeddings, peak_embeddings, tf_id, protein_mask=None, return_all=False,
                return_attention=False):
        tf_tok = self.tf_proj(tf_embeddings, src_key_padding_mask=protein_mask)
        dna_tok = self.encode_dna(peak_embeddings)

        if self.dna_encoding_method == 'onehot':
            dna_tok = self.dna_pos_enc(dna_tok)

        tf_att, attn_tf_to_dna = self.cross_layer_tf(
            tf_tok, dna_tok, query_mask=protein_mask, need_weights=return_attention,
        )

        dna_att, attn_dna_to_tf = self.cross_layer_dna(
            dna_tok, tf_tok, key_padding_mask=protein_mask, need_weights=return_attention,
        )

        seq = torch.cat([tf_att, dna_att], dim=1)
        enc = self.encoder(seq)
        enc_mean, _, _ = self.pool(enc)

        B = enc_mean.size(0)
        logits = torch.zeros(B, 1, device=enc_mean.device)
        for head_idx in range(self.num_tfs):
            mask = (tf_id == head_idx)
            if mask.any():
                logits[mask] = self.tf_specific_heads[head_idx](enc_mean[mask])

        if return_attention:
            attn_dict = {'tf_to_dna': attn_tf_to_dna, 'dna_to_tf': attn_dna_to_tf}
            if return_all:
                shared_logits = self.shared_head(enc_mean)
                return logits, shared_logits, attn_dict
            return logits, attn_dict

        if return_all:
            shared_logits = self.shared_head(enc_mean)
            return logits, shared_logits
        return logits


def create_model(protein_model_name, dna_encoding_method="onehot", dna_max_length=None,
                 proj_type="transformer", num_tfs=10):
    tf_dim = PROTEIN_EMBEDDING_DIMS.get(protein_model_name, 320)

    if dna_encoding_method == "onehot":
        peak_dim = 5
        if dna_max_length is None:
            dna_max_length = 201
    else:
        peak_dim = 512
        if dna_max_length is None:
            dna_max_length = 40

    return AttachedModel(
        tf_input_dim=tf_dim, peak_input_dim=peak_dim,
        dna_encoding_method=dna_encoding_method,
        d_model=512, dim_feedforward=512, nhead=8,
        num_encoder_layers=4, dna_max_length=dna_max_length,
        proj_type=proj_type, num_tfs=num_tfs,
    )
