import os
from dataclasses import dataclass

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


@dataclass
class TokenizerConfig:
    vocab_size: int = 10000
    tokenizer_type: str = "BPE"
    module_path_dir: str = "tokenizers/"


class BPETokenizer:
    def __init__(self, config: TokenizerConfig):
        self.config = config
        self.vocab_size = config.vocab_size
        self._loaded = False

        if config.tokenizer_type.lower() == "bpe":
            special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[SOS]", "[EOS]"]
            self.trainer = BpeTrainer(
                vocab_size=config.vocab_size,
                special_tokens=special_tokens,
            )
            self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
            self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
            self.tokenizer.decoder = ByteLevelDecoder()
        else:
            raise ValueError(f"Unsupported tokenizer type: {config.tokenizer_type}")
        self.load()

    def _iter_corpus(self, batch_size: int = 1000):
        """Stream dataset lazily — never loads the full column into RAM."""
        dataset = load_dataset("roneneldan/TinyStories", streaming=True)
        for split in ("train", "validation"):
            batch = []
            for example in tqdm(dataset[split], desc=f"Streaming {split}"):
                batch.append(example["text"])
                if len(batch) >= batch_size:
                    yield from batch
                    batch.clear()
            if batch:
                yield from batch

    def train(self):
        """Train directly from a streaming iterator — no corpus file needed."""
        self.tokenizer.train_from_iterator(self._iter_corpus(), trainer=self.trainer)
        print("Finished training")

        os.makedirs(self.config.module_path_dir, exist_ok=True)
        save_path = os.path.join(self.config.module_path_dir, "tokenizer.json")
        self.tokenizer.save(save_path)
        print("Saved tokenizer to", save_path)
        self._loaded = True

    def _ensure_loaded(self):
        if self._loaded:
            return
        save_path = os.path.join(self.config.module_path_dir, "tokenizer.json")
        if os.path.exists(save_path):
            print("Loading tokenizer from", save_path)
            self.tokenizer = Tokenizer.from_file(save_path)
            self._loaded = True
        else:
            print("Tokenizer not found. Training...")
            self.train()
            # raise FileNotFoundError(
            #     "Tokenizer not found. Please train the tokenizer first."
            # )

    @staticmethod
    def get_special_tokens() -> dict[str, int]:
        return {
            "[UNK]": 0,
            "[CLS]": 1,
            "[SEP]": 2,
            "[PAD]": 3,
            "[MASK]": 4,
            "[SOS]": 5,
            "[EOS]": 6,
        }

    def load(self):
        self._ensure_loaded()


    def encode(self, text: str) -> list[int]:
        self._ensure_loaded()
        return self.tokenizer.encode(text).ids

    def decode(self, tokens: list[int], format_result: bool = True) -> str:
        self._ensure_loaded()
        result = self.tokenizer.decode(tokens, skip_special_tokens=False)
        if not format_result:
            return result
        return result.replace(" ", "").replace("\n", "")


# if __name__ == "__main__":
#     config = TokenizerConfig()
#     print("Config:", config)

#     tokenizer = BPETokenizer(config)
#     # tokenizer.train()

#     ids = tokenizer.encode("[SOS]Hello world, my name is Jean! not jean but Jean[EOS]")
#     print("Encoded:", ids)
#     print("Decoded:", tokenizer.decode(ids))
#     print("-"*50)
#     ids = tokenizer.encode("[SOS][EOS]")
#     print("Encoded:", ids)
#     print("Decoded:", tokenizer.decode(ids))
