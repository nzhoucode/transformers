# Mini GPT and BERT

This repository contains educational, from-scratch implementations of Transformer-based language model architectures using PyTorch.

It includes:
- A decoder-only **GPT** for autoregressive language modeling.
- An encoder-only **BERT** for masked language modeling.
- Tuned hyperparameters based on the TinyStories training dataset.
- Downscaled to run locally on a personal device.
- Clear and comprehensive documentation.


## Features

### GPT (Generative Pre-trained Transformer)
- Causal multi-head self-attention.
- Rotary positional embedding (RoPE) and scaled dot-product attention for query-key.
- Byte Pair Encoding (BPE) tokenization, ~10K vocabulary size.
- Text generation with temperature and top-p (nucleus) sampling.
- Training loop < 30 minutes, achieves a validation cross-entropy loss < 2.
- ~17M parameters, training loop processes ~40M tokens.

### BERT (Bidirectional Encoder Representations from Transformers)
- Bidirectional multi-head self-attention.
- WordPiece tokenization, ~10K vocabulary size.
- Masked token guess.
- Training loop < 30 minutes, achieves a validation cross-entropy loss < 2.
- ~7M parameters, training loop processes ~150M tokens.
