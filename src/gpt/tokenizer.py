from tokenizers import Tokenizer, models, processors
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
import torch

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer using Hugging Face tokenizers library
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.unk_token = "<|unk|>"
        self.endoftext_token = "<|endoftext|>"
        self.unk_id = None
        self.endoftext_id = None

    def train(self, input_path):
        self.tokenizer = Tokenizer(models.BPE(unk_token=self.unk_token))
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[self.unk_token, self.endoftext_token]
        )
        self.tokenizer.train([input_path], trainer)

        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self.endoftext_id = self.tokenizer.token_to_id(self.endoftext_token)

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single="$A",
            pair="$A $B",
            special_tokens=[(self.endoftext_token, self.endoftext_id)]
        )

    def encode(self, text):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        return self.tokenizer.decode(ids, skip_special_tokens=True)
    
    def encode_file(self, path):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained or loaded.")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = text.replace('\n', ' ')
        chunks = text.split(self.endoftext_token)
        chunks = [chunk.strip() + self.endoftext_token for chunk in chunks if chunk.strip()]
        encodings = self.tokenizer.encode_batch(chunks)
        token_ids = []
        for enc in encodings:
            token_ids.extend(enc.ids)
        return torch.tensor(token_ids, dtype=torch.long)

    def save(self, filepath):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        self.tokenizer.save(filepath)

    def load(self, filepath):
        self.tokenizer = Tokenizer.from_file(filepath)
        self.unk_token = "<|unk|>"
        self.endoftext_token = "<|endoftext|>"
        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self.endoftext_id = self.tokenizer.token_to_id(self.endoftext_token)