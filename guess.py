import torch
from src.bert.transformer import BERT
from src.bert.tokenizer import WordPieceTokenizer

vocab_size = 10000
context_length = 128
dim = 256
depth = 3
heads = 8
ff_dim = 1024
num_segments = 1

# Load tokenizer and model
tokenizer = WordPieceTokenizer(vocab_size)
tokenizer.load('data/bert/tokenizer.json')
pad_id = tokenizer.pad_id
mask_id = tokenizer.mask_id

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = BERT(vocab_size, context_length, dim, depth, heads, ff_dim, num_segments).to(device)
model.load_state_dict(torch.load('bertmodel.pt', map_location=device))
model.eval()

# Accept user input
prompt = input("Enter sentence with [MASK] for fill-in-the-blank (case-insensitive): ")
input_ids = tokenizer.encode(prompt)
if len(input_ids) > context_length:
    raise ValueError("Input exceeds context length")
try:
    mask_index = input_ids.index(mask_id)
except ValueError:
    raise ValueError("Input must contain [MASK]")

x = torch.tensor([input_ids], dtype=torch.long).to(device)

# Guess masked token
padding_mask = (x != pad_id).long()  # shape: (B, T)
with torch.no_grad():
    logits = model(x, segment_ids=None, padding=padding_mask)
    predicted_token_id = torch.argmax(logits[0, mask_index]).item()
x[0, mask_index] = predicted_token_id
print(tokenizer.decode(x[0].tolist()))