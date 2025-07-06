import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    """ 
    Bidirectional multi-head self-attention layer
    Each head processes a different subspace of the embedding.
    Computes attention scores between tokens using query-key dot-products.
    Enables contextualization of tokens.
    """
    def __init__(self, dim, heads, dropout=0):
        super().__init__()
        assert dim % heads == 0, "Embedding dimension must be divisible by number of heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, padding=None):
        B, T, C = x.shape  # batch size, sequence length, embedding dimension
        qkv = self.qkv(x).chunk(3, dim=-1)  # query, key, value tensors

        # Reshape into multiple heads
        q, k, v = [
            t.view(B, T, self.heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
            for t in qkv
        ]
        # Scaled dot-product attention for query-key similarity of every token
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        # No causal mask
        # Padding attention mask
        if padding is not None:
            padding = padding.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(padding == 0, float('-inf'))  # (B, 1, 1, T)
        
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
    def __init__(self, dim, heads, ff_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, heads)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)

    def forward(self, x, padding=None):
        x = x + self.attn(self.ln1(x), padding)
        x = x + self.ff(self.ln2(x))
        return x