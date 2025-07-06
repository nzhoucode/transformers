import torch
import torch.nn as nn
from src.bert.modules import TransformerBlock

class BERT(nn.Module):
    """
    Encoder-only bidirectional transformer (BERT) for masked language modeling
    """
    def __init__(self, vocab_size, context_length, dim, depth, heads, ff_dim, num_segments):
        super().__init__()
        # Token and positional embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, context_length, dim))
        self.seg_emb = nn.Embedding(num_segments, dim)
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ff_dim) for _ in range(depth)]
        )
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Maximum context length
        self.context_length = context_length

    def forward(self, x, segment_ids=None, padding=None):
        B, T = x.size()
        assert T <= self.context_length
        if segment_ids is None:
            segment_ids = torch.zeros_like(x)
        x = self.token_emb(x) + self.pos_emb[:, :T, :] + self.seg_emb(segment_ids)  # (B, T, dim)
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, padding)
        # Final layer norm and projection to vocabulary logits
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)