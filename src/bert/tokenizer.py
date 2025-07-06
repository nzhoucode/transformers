from tokenizers import Tokenizer, models, trainers, processors
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence, Replace
from tokenizers.decoders import WordPiece as WordPieceDecoder
import torch

class WordPieceTokenizer:
    """
    WordPiece tokenizer using Hugging Face tokenizers library
    """
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = None
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.endoftext_token = "<|endoftext|>"
        self.pad_id = None
        self.unk_id = None
        self.cls_id = None
        self.sep_id = None
        self.mask_id = None
        self.endoftext_id = None

    def train(self, input_path):
        self.tokenizer = Tokenizer(models.WordPiece(unk_token=self.unk_token))
        self.tokenizer.normalizer = Sequence([
            NFD(), 
            Lowercase(), 
            StripAccents(),
        ])
        self.tokenizer.pre_tokenizer = Whitespace()

        trainer = trainers.WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=[
                self.pad_token,
                self.unk_token,
                self.cls_token,
                self.sep_token,
                self.mask_token,
                self.endoftext_token
            ],
        )
        self.tokenizer.train([input_path], trainer)

        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self.cls_id = self.tokenizer.token_to_id(self.cls_token)
        self.sep_id = self.tokenizer.token_to_id(self.sep_token)
        self.mask_id = self.tokenizer.token_to_id(self.mask_token)
        self.endoftext_id = self.tokenizer.token_to_id(self.endoftext_token)

        self.tokenizer.decoder = WordPieceDecoder(prefix="##")

        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.cls_token} $A {self.sep_token}",
            pair=f"{self.cls_token} $A {self.sep_token} $B {self.sep_token}",
            special_tokens=[
                (self.cls_token, self.cls_id),
                (self.sep_token, self.sep_id)
            ]
        )

    def encode(self, text, pair=None):
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained.")
        return self.tokenizer.encode(text, pair).ids

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
        chunks = [chunk.strip() + f' {self.endoftext_token}' for chunk in chunks if chunk.strip()]
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
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        self.endoftext_token = "<|endoftext|>"
        self.pad_id = self.tokenizer.token_to_id(self.pad_token)
        self.unk_id = self.tokenizer.token_to_id(self.unk_token)
        self.cls_id = self.tokenizer.token_to_id(self.cls_token)
        self.sep_id = self.tokenizer.token_to_id(self.sep_token)
        self.mask_id = self.tokenizer.token_to_id(self.mask_token)
        self.endoftext_id = self.tokenizer.token_to_id(self.endoftext_token)
        self.tokenizer.decoder = WordPieceDecoder(prefix="##")
