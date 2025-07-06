import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary positional embedding (RoPE)
    Rotates query and key embeddings by positional encoding.
    """
    def __init__(self, head_dim, max_len=2048, base=10000):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        self.head_dim = head_dim
        self.max_len = max_len  # max sequence length
        half_dim = head_dim // 2
        theta = 1.0 / (base ** (torch.arange(half_dim) / half_dim))  # (half_dim,)
        pos = torch.arange(max_len).float()  # (T,)
        angles = torch.outer(pos, theta)  # (T, half_dim)

        sin = angles.sin()[None, None, :, :]  # (1, 1, T, half_dim)
        cos = angles.cos()[None, None, :, :]  # (1, 1, T, half_dim)
        self.register_buffer("sin_cached", sin, persistent=False)
        self.register_buffer("cos_cached", cos, persistent=False)

    def forward(self, x, seq_start=0):
        T = x.size(-2)
        sin = self.sin_cached[:, :, seq_start:seq_start + T, :]
        cos = self.cos_cached[:, :, seq_start:seq_start + T, :]
        x1 = x[..., ::2]   # even-indexed dims
        x2 = x[..., 1::2]  # odd-indexed dims
        # (B, heads, T, half_dim)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # (B, heads, T, head_dim)

class MultiHeadSelfAttention(nn.Module):
    """ 
    Causal multi-head self-attention layer
    Each head processes a different subspace of the embedding.
    Computes attention scores between tokens using RoPE and query-key dot-products.
    Enables contextualization of tokens autoregressively, does not attend to future tokens.
    """
    def __init__(self, dim, heads, context_length, dropout=0):
        super().__init__()
        assert dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

        self.rope = RoPE(head_dim=self.head_dim, max_len=2*context_length)
    
    def forward(self, x, pos=0):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        qkv = self.qkv(x).chunk(3, dim=-1)  # query, key, value tensors

        # Reshape into multiple heads
        q, k, v = [
            t.view(B, T, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
            for t in qkv
        ]

        # RoPE for query and key
        q = self.rope.forward(q, seq_start=pos)
        k = self.rope.forward(k, seq_start=pos)

        # Scaled dot-product attention for query-key similarity of every token
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        # Causal mask for autoregression
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        attn = attn.masked_fill(causal_mask == 0, float('-inf'))

        # Softmax attention into weights
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention weights to value vectors
        out = (attn @ v)  # (B, heads, T, head_dim)
        # Concatenate and combine head outputs
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        return self.out_dropout(self.proj(out))

class FeedForward(nn.Module):
    """
    Position-wise feedforward layer
    Applies two linear layers with GELU activation and optional dropout.
    """
    def __init__(self, dim, hidden_dim, dropout=0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """
    Transformer block
    Includes multi-head self-attention and feedforward layers,
    each with pre-layer normalization and followed by residual connection.
    """
    def __init__(self, dim, heads, ff_dim, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads, context_length)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)

    def forward(self, x, pos=0):
        x = x + self.attn(self.ln1(x), pos=pos)
        x = x + self.ff(self.ln2(x))
        return x