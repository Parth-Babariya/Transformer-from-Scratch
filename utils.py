# utils.py
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[0]
        if self.cos_cached is None or seq_len > self.cos_cached.shape[-2]:
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.cos_cached = emb.cos()[None, None, :, :]
            self.sin_cached = emb.sin()[None, None, :, :]
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, pe_strategy=None, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.pe_strategy = pe_strategy

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None, q_rotary_cos=None, q_rotary_sin=None, k_rotary_cos=None, k_rotary_sin=None):
        q_seq_len, batch_size, _ = Q.shape
        k_seq_len, _, _ = K.shape
        v_seq_len, _, _ = V.shape

        q_s = self.W_q(Q).view(q_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        k_s = self.W_k(K).view(k_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        v_s = self.W_v(V).view(v_seq_len, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        
        if self.pe_strategy == 'rope' and q_rotary_cos is not None and k_rotary_cos is not None:
            q_s = apply_rotary_pos_emb(q_s, q_rotary_cos, q_rotary_sin)
            k_s = apply_rotary_pos_emb(k_s, k_rotary_cos, k_rotary_sin)

        scores = torch.matmul(q_s, k_s.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, v_s)

        context = context.permute(2, 0, 1, 3).contiguous().view(q_seq_len, batch_size, self.d_model)
        output = self.W_o(context)
        return output

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.relu(self.linear1(x))))

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, tgt, src_mask, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        memory = self.encode(src, src_mask)
        output = self.decode(memory, tgt, src_mask, tgt_mask)
        return self.generator(output)

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.proj(x)