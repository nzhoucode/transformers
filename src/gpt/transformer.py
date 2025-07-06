import torch
import torch.nn as nn
from src.gpt.modules import TransformerBlock

class GPT(nn.Module):
    """
    Decoder-only generative pre-trained transformer (GPT) for autoregressive language modeling
    """
    def __init__(self, vocab_size, context_length, dim, depth, heads, ff_dim):
        super().__init__()
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, dim)
        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, ff_dim, context_length) for _ in range(depth)]
        )
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Maximum context length
        self.context_length = context_length

    def forward(self, x, pos=0):
        B, T = x.size()
        assert T <= self.context_length
        x = self.token_emb(x)  # (B, T, dim)
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, pos=pos)
        # Final layer norm and projection to vocabulary logits
        x = self.ln_f(x)
        return self.head(x)  # (B, T, vocab_size)

    def generate(self, tokens, num_new, top_p=0.9, temperature=1.0, endoftext_id=None):
        for _ in range(num_new):
            context = tokens[:, -self.context_length:]
            pos = tokens.size(1) - context.size(1)
            logits = self(context, pos=pos)  # (B, T, V)
            logits = logits[:, -1, :]  # (B, V)

            # Apply temperature
            logits = logits / temperature

            # Top-p (nucleus) filtering
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            # Mask tokens with cumulative prob > top_p
            sorted_mask = cumulative_probs > top_p
            sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
            sorted_mask[..., 0] = False  # always keep at least 1 token
            sorted_logits[sorted_mask] = float('-inf')
            # Scatter filtered logits back to original order
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(1, sorted_indices, sorted_logits)

            # Sample from filtered distribution
            probs = torch.softmax(logits_filtered, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

            # Break if end
            if next_token[0, 0].item() == endoftext_id:
                break

        return tokens