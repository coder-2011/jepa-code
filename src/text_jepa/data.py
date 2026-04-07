import json
import random
from pathlib import Path

import torch

from .batching import collate_masked_examples
from .masking import mask_text_from_yaml


class FineWebJsonlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        jsonl_path,
        tokenizer,
        config_path,
        seed=0,
        min_token_count=0,
        min_language_score=None,
        max_docs=None,
    ):
        self.tokenizer = tokenizer
        self.config_path = config_path
        self.seed = seed
        self.texts = []

        path = Path(jsonl_path)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                text = (row.get("text") or "").strip()
                if not text:
                    continue

                token_count = row.get("token_count")
                # Missing token_count is only tolerated when the caller does not request a lower bound.
                if isinstance(token_count, int) and token_count < min_token_count:
                    continue
                if min_token_count > 0 and not isinstance(token_count, int):
                    continue

                language_score = row.get("language_score")
                if min_language_score is not None:
                    # Keep filtering strict so low-confidence language rows never silently pass through.
                    if not isinstance(language_score, (int, float)) or language_score < min_language_score:
                        continue

                self.texts.append(text)
                if max_docs is not None and len(self.texts) >= max_docs:
                    break

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        # Seed per example so dataloader shuffling does not change which spans get masked.
        rng = random.Random(self.seed + index)
        return mask_text_from_yaml(
            self.tokenizer,
            self.texts[index],
            self.config_path,
            rng=rng,
        )


def create_fineweb_dataloader(
    jsonl_path,
    tokenizer,
    config_path,
    batch_size,
    shuffle=False,
    seed=0,
    num_workers=0,
    min_token_count=0,
    min_language_score=None,
    max_docs=None,
    drop_last=False,
):
    # Dataset construction owns text filtering; DataLoader owns shuffling and batch assembly.
    dataset = FineWebJsonlDataset(
        jsonl_path=jsonl_path,
        tokenizer=tokenizer,
        config_path=config_path,
        seed=seed,
        min_token_count=min_token_count,
        min_language_score=min_language_score,
        max_docs=max_docs,
    )
    generator = None
    if shuffle:
        # Seed the loader generator so document order changes are reproducible across runs.
        generator = torch.Generator()
        generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_masked_examples,
        generator=generator,
    )
