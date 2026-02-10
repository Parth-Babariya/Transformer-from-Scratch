import torch.nn as nn
from utils import MultiHeadAttention, PositionwiseFeedForward, RotaryPositionalEmbeddings

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, pe_strategy):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, pe_strategy, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, pe_strategy, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask, tgt_mask, rotary_tgt_cos=None, rotary_tgt_sin=None, rotary_src_cos=None, rotary_src_sin=None):
        attn_output = self.self_attn(x, x, x, tgt_mask, rotary_tgt_cos, rotary_tgt_sin, rotary_tgt_cos, rotary_tgt_sin)
        x = self.norm1(x + self.dropout(attn_output))
        
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask, rotary_tgt_cos, rotary_tgt_sin, rotary_src_cos, rotary_src_sin)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class Decoder(nn.Module):
    def __init__(self, layer, N, pe_strategy):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(N)])
        self.norm = nn.LayerNorm(layer.self_attn.d_model)
        self.pe_strategy = pe_strategy
        if self.pe_strategy == 'rope':
            self.rope = RotaryPositionalEmbeddings(dim=layer.self_attn.d_k)
            
    def forward(self, x, memory, src_mask, tgt_mask):
        rotary_tgt_cos, rotary_tgt_sin = (None, None)
        rotary_src_cos, rotary_src_sin = (None, None)
        if self.pe_strategy == 'rope':
            rotary_tgt_cos, rotary_tgt_sin = self.rope(x)
            rotary_src_cos, rotary_src_sin = self.rope(memory)
            
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, rotary_tgt_cos, rotary_tgt_sin, rotary_src_cos, rotary_src_sin)
        return self.norm(x)