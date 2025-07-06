import os
import math
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR
from src.bert.transformer import BERT
from src.bert.tokenizer import WordPieceTokenizer

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameter config
vocab_size = 10000
context_length = 128
dim = 256
depth = 3
heads = 8
ff_dim = 1024
num_segments = 1
batch_size = 128

# Tokenize data
tokenizer_path = 'data/bert/tokenizer.json'
tokenizer = WordPieceTokenizer(vocab_size)
if os.path.exists(tokenizer_path):
    print("Loading tokenizer...")
    tokenizer.load(tokenizer_path)
else:
    print("Training tokenizer...")
    tokenizer.train('data/bert/train.txt')  # ASCII-only
    tokenizer.save(tokenizer_path)
pad_id = tokenizer.pad_id
mask_id = tokenizer.mask_id

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
train_data = encode_data('data/bert/train.txt', 'data/bert/train.pt')
val_data = encode_data('data/valid.txt', 'data/bert/valid.pt')

# Batch sampling
def get_batch(data_array, batch_size, context_length, device):
    max_start = len(data_array) - context_length
    assert max_start > 0, "Dataset too small"
    starts = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data_array[i:i+context_length].clone() for i in starts])
    return x.to(device)

# Masked language modeling transformation
def create_mlm_inputs(batch, mlm_prob=0.15):
    y = batch.clone()  # labels
    prob_matrix = torch.full(y.shape, mlm_prob, device=device)
    pad_mask = batch == pad_id
    prob_matrix.masked_fill_(pad_mask, value=0.0)
    mask = torch.bernoulli(prob_matrix).bool()  # sample mask
    mask &= ~pad_mask
    y[~mask] = -100  # ignore non-masked for loss

    x = batch.clone()  # inputs
    rand = torch.rand(batch.shape, device=batch.device)
    # Replace 80% of masked tokens with [MASK]
    mask_80 = (rand < 0.8) & mask
    x[mask_80] = mask_id
    # Leave 20% of masked tokens unchanged
    # Note: larger BERT does 10% random token, 10% unchanged
    return x, y

# Instantiate model, optimizer, and learning rate scheduler
model = BERT(vocab_size, context_length, dim, depth, heads, ff_dim, num_segments).to(device)
base_lr = 1e-3
optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
total_steps = 10000
warmup_steps = 400  # 4%
anneal_steps = 9000  # 90%
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
        val_batch = get_batch(val_data, batch_size, context_length, device)
        val_x, val_y = create_mlm_inputs(val_batch)
        val_padding = (val_batch != pad_id)
        val_logits = model(val_x, segment_ids=None, padding=val_padding)
        val_loss = F.cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1), ignore_index=-100)
        print(f"Step {step}, Validation Loss: {val_loss.item():.4f}")
    model.train()

# Training loop
print("Training BERT...")
for step in range(total_steps):
    model.train()
    # Get masked inputs and labels
    batch = get_batch(train_data, batch_size, context_length, device)
    x, y = create_mlm_inputs(batch)
    padding = (batch != pad_id)
    
    # Forward pass and loss
    optimizer.zero_grad()
    logits = model(x, segment_ids=None, padding=padding)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

    # Backpropagation
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    if step % 100 == 0:
        print(f"Step {step}, Training Loss: {loss.item():.4f}")
        if step % 500 == 0:
            validation(step)

validation(total_steps)

# Save checkpoint
torch.save(model.state_dict(), 'bertmodel.pt')
