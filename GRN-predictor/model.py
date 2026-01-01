import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
class ProteinTFProjection(nn.Module):
    """
    Transformer-based projection head for ESM2 protein embeddings.
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
        x = self.input_proj(protein_emb)          
        x = self.norm_in(x)
        x = self.transformer(x)                   
        x = self.final_norm(x)
        
        return x


class CrossAttentionLayer(nn.Module):
    """
    corss attention layer
    """
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
        raw_scores = self.score(x).squeeze(-1)      
        weights = torch.softmax(raw_scores, dim=-1) 

        pooled = torch.bmm(weights.unsqueeze(1), x).squeeze(1) 

        return pooled, raw_scores, weights   

class AttachedModel(nn.Module):
    """
    Improved Protein-DNA interaction model with bidirectional cross-attention.
    """
    def __init__(
        self,
        tf_input_dim: int = 320,
        peak_input_dim: int = 5,
        d_model: int = 512,
        dna_encoding_method: str = "onehot",
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        protein_length: int = 903,
        dna_max_length: int = 40, #Nucleotide transformer embedding length
        dropout: float = 0.1
    ):
        super().__init__()
        if dna_encoding_method == "onehot": 
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

        # Bidirectional cross-attention with full layers
        
        self.cross_layer_tf = CrossAttentionLayer(d_model, nhead, dim_feedforward, 0.4) #
        self.cross_layer_dna = CrossAttentionLayer(d_model, nhead, dim_feedforward, 0.4)
        
        self.pool=AttentivePoolWithScores(self.d_model)

        # Classifier: Simplified, with pooled DNA input
        self.classifier = nn.Sequential(
            nn.Linear(self.dna_max_length + self.protein_length, 256), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def encode_dna(self, peak_embeddings: torch.Tensor) -> torch.Tensor:
        """
        depends on the dna embedding methods
        """
        if peak_embeddings.dim() == 3 and peak_embeddings.size(-1) == 5:
            x = peak_embeddings.transpose(1, 2)
            x = self.dna_conv(x)
            x = x.transpose(1, 2)
            return x
        else:
            return peak_embeddings

    def forward(self, tf_embeddings: torch.Tensor, peak_embeddings: torch.Tensor, sigmoid: bool = True):
       

        # TF token
        tf_tok = self.tf_proj(tf_embeddings)
        if self.dna_encoding_method=='onehot':
            # DNA tokens
            dna_tok = peak_embeddings
        elif self.dna_encoding_method=='transformer':
            dna_tok = self.encode_dna(peak_embeddings) 

        tf_att, _   = self.cross_layer_tf(tf_tok, dna_tok)     
        dna_att, _  = self.cross_layer_dna(dna_tok, tf_tok)     

        

        # Concatenate along sequence dimension
        seq = torch.cat([tf_att, dna_att], dim=1) 

        
        # Joint self-attention over BOTH protein and DNA residues
        enc = self.encoder(seq)                                
        enc_mean = enc.mean(dim=-1)     
        # Classifier with proper dimension
        logits = self.classifier(enc_mean).squeeze(-1)   # (B,)

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
