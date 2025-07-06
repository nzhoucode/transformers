import torch
from src.gpt.transformer import GPT
from src.gpt.tokenizer import BPETokenizer

vocab_size = 10000
context_length = 256
dim = 512
depth = 4
heads = 16
ff_dim = 1344

# Load tokenizer and model
tokenizer = BPETokenizer()
tokenizer.load('data/gpt/tokenizer.json')

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = GPT(vocab_size, context_length, dim, depth, heads, ff_dim).to(device)
model.load_state_dict(torch.load('gptmodel.pt', map_location=device))
model.eval()

# Accept user input
prompt = input("Enter prompt: ")
input_ids = tokenizer.encode(prompt)
x = torch.tensor([input_ids], dtype=torch.long).to(device)

# Trim to fit
if x.size(1) > model.context_length:
    x = x[:, -model.context_length:]

# Generate tokens
with torch.no_grad():
    out = model.generate(x, num_new=256, top_p=0.9, temperature=0.8, endoftext_id=1)
    print(tokenizer.decode(out[0].tolist()))