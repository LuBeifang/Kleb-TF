import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
class ProteinTFProjection(nn.Module):
    """
    Transformer-based projection head for ESM2 protein embeddings.
    Input : (B, L_p=1024, 320)   → ESM2 per-residue embeddings
    Output: (B, L_p=1024, 512)   → ready for cross-attention with DNA (B, 201, 512)
    """
    def __init__(
        self,
        input_dim: int = 320,
        output_dim: int = 512,
        n_layers: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, output_dim)   # 320 → 512
        self.norm_in = nn.LayerNorm(output_dim)

        # Learnable residue-type positional encoding (since ESM2 already has structure info)
        #self.pos_encoding = nn.Parameter(torch.zeros(1, 1024, output_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dim,
            nhead=n_heads,
            dim_feedforward=output_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.final_norm = nn.LayerNorm(output_dim)

    def forward(self, protein_emb: torch.Tensor) -> torch.Tensor:
        """
        protein_emb: (B, 1024, 320)
        returns    : (B, 1024, 512)
        """
        x = self.input_proj(protein_emb)          # (B, 1024, 512)
        x = self.norm_in(x)
        x = self.transformer(x)                   # self-attention over 1024 residues
        x = self.final_norm(x)
        
        return x


class CrossAttentionLayer(nn.Module):
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
        query: torch.Tensor,              # [B, Tq, D]
        key_value: torch.Tensor,          # [B, Tk, D]
        attn_mask: Optional[torch.Tensor] = None,         # [Tq, Tk] or [B*nhead, Tq, Tk]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, Tk] (True for padding)
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_w = self.attn(
            query, key_value, key_value,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
        )
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return (x, attn_w) if need_weights else (x, None)
class AttentivePoolWithScores(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, L, D)
        raw_scores = self.score(x).squeeze(-1)      # (B, L)    未归一化分数
        weights = torch.softmax(raw_scores, dim=-1) # (B, L)    注意力权重

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1)  # (B, D)

        return pooled, raw_scores, weights   # (B, D), (B, L), (B, L)


class AttachedModel(nn.Module):
    """
    Improved Protein-DNA interaction model with bidirectional cross-attention using full transformer-like layers (attention + FFN + norms).
    """
    def __init__(
        self,
        tf_input_dim: int = 320,
        peak_input_dim: int = 5,  # For one-hot
        d_model: int = 512,
        dna_encoding_method: str = "onehot",
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        protein_length: int = 903,
        dna_max_length: int = 40 #kmer 40x6mer
        #dropout: float = 0.1 var for each block
    ):
        super().__init__()
        if dna_encoding_method == "onehot":  # Focused on onehot for this improvement
            self.dna_encoding_method='onehot'
            self.peak_input_dim = 5
        else:
            self.dna_encoding_method='transformer'
            self.peak_input_dim = 512
        self.d_model = d_model
        self.protein_length = protein_length
        self.dna_max_length = dna_max_length
        # TF projection
        self.tf_proj =ProteinTFProjection(
                            input_dim=320,
                            output_dim=512,
                            n_layers=4,
                            n_heads=8,
                            dropout=0.3
                        )
        # DNA CNN without residuals
        self.dna_conv = nn.Sequential(
            nn.Conv1d(self.peak_input_dim, 128, kernel_size=12, padding=6),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 384, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Conv1d(384, d_model, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(0.2),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)


        # #bidirectional LSTM for DNA feature concentraion
        self.lstm_dna = nn.LSTM(
            input_size=512,
            hidden_size=256,
            num_layers=2,
            batch_first=True,     # (B, T, D)
            bidirectional=True,
            dropout=0.2
        )
        # Bidirectional cross-attention with full layers
        
        self.cross_layer_tf = CrossAttentionLayer(d_model, nhead, dim_feedforward, 0.4) #
        self.cross_layer_dna = CrossAttentionLayer(d_model, nhead, dim_feedforward, 0.4)

        self.pool=AttentivePoolWithScores(self.d_model)

        # Classifier: Simplified, with pooled DNA input
        self.classifier = nn.Sequential(
            nn.Linear(self.dna_max_length + self.protein_length, 256),  # TF + DNA avg + DNA max
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),  # TF + DNA avg + DNA max
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def encode_dna(self, peak_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns DNA tokens as (B, L', D); L=220 -> L' ~28 after downsampling.
        Args:
            peak_embeddings: Tensor of shape (B, L, 5) for one-hot or (B, L', D) for embeddings.
        """
        # Case 1: one-hot DNA, shape (B, L, 5)
        if peak_embeddings.dim() == 3 and peak_embeddings.size(-1) == 5:
            # (B, L, 5) -> (B, 5, L)
            x = peak_embeddings.transpose(1, 2)
            # (B, 5, L) -> (B, D, L')
            x = self.dna_conv(x)
            # (B, D, L') -> (B, L', D)
            x = x.transpose(1, 2)
            return x
        # Case 2: already embedded, e.g., (B, L', D)
        else:
            # # (B, L'), 直接投影到 (B, L', D)，假设L'=512
            # # 你可以用一个Linear层或者expand
            # x = peak_embeddings.unsqueeze(-1)  # (B, L', 1)
            # x = x.expand(-1, -1, self.d_model) # (B, L', D)
            return peak_embeddings
        # else:
        #     raise ValueError(f"Unexpected peak_embeddings shape: {peak_embeddings.shape}")
    def forward(self, tf_embeddings: torch.Tensor, peak_embeddings: torch.Tensor, sigmoid: bool = True):
        #B = tf_embeddings.size(0)

        # TF token
        tf_tok = self.tf_proj(tf_embeddings)
        if self.dna_encoding_method=='onehot':
            # DNA tokens
            dna_tok = peak_embeddings
        elif self.dna_encoding_method=='transformer':
            dna_tok = self.encode_dna(peak_embeddings) 


        #dna_tok_lstm,_=self.lstm_dna(dna_tok)
        # # Bidirectional cross-attention with full layers
        tf_att, _   = self.cross_layer_tf(tf_tok, dna_tok)      # TF attends to DNA → (B, 1024, 512)
        dna_att, _  = self.cross_layer_dna(dna_tok, tf_tok)     # DNA attends to TF → (B, 201, 512)

        

        # Concatenate along sequence dimension
        seq = torch.cat([tf_att, dna_att], dim=1)   # → (B, 1024 + 201 = 1225, 512)

        
        # Joint self-attention over BOTH protein and DNA residues
        enc = self.encoder(seq)                                # → (B, 1225, 320)
        enc_pooled,enc_score,enc_weigths = self.pool(enc)   #weight out put B L
        # Classifier with proper dimension
        logits = self.classifier(enc_weigths).squeeze(-1)   # (B,)

        if sigmoid:
            return torch.sigmoid(logits)
        return logits   
    
# Updated factory
def create_attached_model(protein_model_name, dna_encoding_method="onehot", dna_max_length=None):
    protein_embedding_dims = {
        "facebook/esm2_t6_8M_UR50D": 320,
        "facebook/esm2_t12_35M_UR50D": 480,
        "facebook/esm2_t30_150M_UR50D": 640,
        "facebook/esm2_t33_650M_UR50D": 1280,
        "Rostlab/prot_bert": 1024,
        "Rostlab/prot_bert_bfd": 1024,
        "Rostlab/prot_t5_xl_uniref50": 1024,
    }
    
    tf_dim = protein_embedding_dims.get(protein_model_name, 320)
    
    if dna_encoding_method == "onehot":
        peak_dim = 5
    else:
        peak_dim=512
    
    return AttachedModel(
        tf_input_dim=tf_dim, 
        peak_input_dim=peak_dim,
        dna_encoding_method=dna_encoding_method,
        d_model=512,
        dim_feedforward=512,
        nhead=8,
        num_encoder_layers=8
    )