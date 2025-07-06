import os
import math
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from src.gpt.transformer import GPT
from src.gpt.tokenizer import BPETokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameter config
vocab_size = 10000
context_length = 256
dim = 512       # embedding size, must be divisible by heads
depth = 4       # number of transformer blocks
heads = 16      # number of attention heads
ff_dim = 1344   # feedforward size
batch_size = 32

# Tokenize data
tokenizer_path = 'data/gpt/tokenizer.json'
tokenizer = BPETokenizer(vocab_size)
if os.path.exists(tokenizer_path):
    print("Loading tokenizer...")
    tokenizer.load(tokenizer_path)
else:
    print("Training tokenizer...")
    tokenizer.train('data/train.txt')
    tokenizer.save(tokenizer_path)

# Encode datasets
def encode_data(path, save_path):
    if os.path.exists(save_path):
        print(f"Loading {save_path}...")
        return torch.load(save_path)
    else:
        print(f"Encoding {path}...")
        data = tokenizer.encode_file(path)
        torch.save(data, save_path)
        return data
train_data = encode_data('data/train.txt', 'data/gpt/train.pt')
val_data = encode_data('data/valid.txt', 'data/gpt/valid.pt')

# Batch sampling
def get_batch(data_array, batch_size, context_length, device):
    max_start = len(data_array) - context_length - 1
    assert max_start > 0, "Dataset too small"

    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_array[i : i + context_length].clone() for i in starts])
    # Target output
    y = torch.stack([data_array[i + 1 : i + context_length + 1].clone() for i in starts])
    return x.to(device), y.to(device)

# Instantiate model, optimizer, and learning rate scheduler
model = GPT(vocab_size, context_length, dim, depth, heads, ff_dim).to(device)
base_lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
total_steps = 5000
warmup_steps = 200  # 4%
anneal_steps = 4500  # 90%
min_lr = 1e-4
def lr_lambda(step):
    if step < warmup_steps:
        return step / warmup_steps
    elif step < anneal_steps:
        progress = (step - warmup_steps) / (anneal_steps - warmup_steps)
        lr = min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (base_lr - min_lr)
        return lr / base_lr  # cosine annealing
    else:
        return min_lr / base_lr
scheduler = LambdaLR(optimizer, lr_lambda)

# Validation function
def validation(step):
    model.eval()
    with torch.no_grad():
        val_x, val_y = get_batch(val_data, batch_size, context_length, device)
        val_logits = model(val_x)
        val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
        print(f"Step {step}, Validation Loss: {val_loss.item():.4f}")
    model.train()

# Training loop
print("Training GPT...")
for step in range(total_steps):
    model.train()
    # Get batch
    x, y = get_batch(train_data, batch_size, context_length, device)

    # Forward pass and loss
    optimizer.zero_grad()
    logits = model(x)  # (B, T, V)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}, Training Loss: {loss.item():.4f}")
        if step % 500 == 0:
            validation(step)

validation(total_steps)

# Save model checkpoint
torch.save(model.state_dict(), 'gptmodel.pt')