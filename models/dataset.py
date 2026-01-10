"""
Dataset and data loading utilities for Malayalam handwriting data.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


class MalayalamVocab:
    """
    Character vocabulary for Malayalam text.
    Handles encoding/decoding of Malayalam characters.
    """

    # Special tokens
    PAD = '<PAD>'
    UNK = '<UNK>'
    SOS = '<SOS>'
    EOS = '<EOS>'

    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self._build_vocab()

    def _build_vocab(self):
        """Build vocabulary from Malayalam Unicode range + special tokens."""
        special_tokens = [self.PAD, self.UNK, self.SOS, self.EOS]

        for idx, token in enumerate(special_tokens):
            self.char_to_idx[token] = idx
            self.idx_to_char[idx] = token

        idx = len(special_tokens)

        # Malayalam Unicode range: U+0D00 to U+0D7F
        for code in range(0x0D00, 0x0D80):
            char = chr(code)
            self.char_to_idx[char] = idx
            self.idx_to_char[idx] = char
            idx += 1

        # Add common punctuation and digits
        for char in '0123456789.,!?-–—\'\"():;':
            if char not in self.char_to_idx:
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
                idx += 1

        # Add space
        if ' ' not in self.char_to_idx:
            self.char_to_idx[' '] = idx
            self.idx_to_char[idx] = ' '

    def encode(self, text: str, add_sos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to list of indices."""
        indices = []

        if add_sos:
            indices.append(self.char_to_idx[self.SOS])

        for char in text:
            indices.append(self.char_to_idx.get(char, self.char_to_idx[self.UNK]))

        if add_eos:
            indices.append(self.char_to_idx[self.EOS])

        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text."""
        chars = []
        for idx in indices:
            char = self.idx_to_char.get(idx, self.UNK)
            if char not in [self.PAD, self.SOS, self.EOS]:
                chars.append(char)
        return ''.join(chars)

    def __len__(self):
        return len(self.char_to_idx)


def normalize_strokes(strokes: np.ndarray, scale_factor: float = None) -> Tuple[np.ndarray, float]:
    """
    Normalize stroke data.

    Args:
        strokes: [seq_len, 5] array
        scale_factor: if provided, use this scale; otherwise compute from data

    Returns:
        normalized_strokes: [seq_len, 5] normalized array
        scale_factor: the scale factor used
    """
    if len(strokes) == 0:
        return strokes, 1.0

    # Extract dx, dy
    offsets = strokes[:, :2]

    if scale_factor is None:
        # Compute scale as std of offsets
        scale_factor = np.std(offsets)
        if scale_factor < 1e-6:
            scale_factor = 1.0

    # Normalize offsets
    normalized = strokes.copy()
    normalized[:, :2] = offsets / scale_factor

    return normalized, scale_factor


def augment_strokes(strokes: np.ndarray, scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
    """
    Apply random augmentation to strokes.

    Args:
        strokes: [seq_len, 5] array
        scale_range: range for random scaling

    Returns:
        augmented strokes
    """
    augmented = strokes.copy()

    # Random scaling
    scale = np.random.uniform(*scale_range)
    augmented[:, :2] *= scale

    # Random small rotation (optional)
    angle = np.random.uniform(-0.1, 0.1)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x, y = augmented[:, 0], augmented[:, 1]
    augmented[:, 0] = cos_a * x - sin_a * y
    augmented[:, 1] = sin_a * x + cos_a * y

    return augmented


class MalayalamHandwritingDataset(Dataset):
    """
    Dataset for Malayalam handwriting samples.
    Loads from JSONL format saved by the collection tool.
    """

    def __init__(
        self,
        data_path: str,
        vocab: MalayalamVocab,
        max_stroke_len: int = 200,
        max_text_len: int = 50,
        normalize: bool = True,
        augment: bool = False
    ):
        """
        Args:
            data_path: path to JSONL file or directory with JSONL files
            vocab: MalayalamVocab instance
            max_stroke_len: maximum stroke sequence length
            max_text_len: maximum text length
            normalize: whether to normalize strokes
            augment: whether to apply data augmentation
        """
        self.vocab = vocab
        self.max_stroke_len = max_stroke_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        self.augment = augment

        self.samples = []
        self._load_data(data_path)

        # Compute global scale factor for normalization
        if normalize and len(self.samples) > 0:
            all_offsets = []
            for sample in self.samples:
                strokes = np.array(sample['stroke3'], dtype=np.float32)
                if len(strokes) > 0:
                    all_offsets.append(strokes[:, :2])

            if all_offsets:
                all_offsets = np.concatenate(all_offsets, axis=0)
                self.scale_factor = np.std(all_offsets)
                if self.scale_factor < 1e-6:
                    self.scale_factor = 1.0
            else:
                self.scale_factor = 1.0
        else:
            self.scale_factor = 1.0

    def _load_data(self, data_path: str):
        """Load samples from JSONL file(s)."""
        path = Path(data_path)

        if path.is_file():
            self._load_jsonl(path)
        elif path.is_dir():
            for jsonl_file in path.glob('*.jsonl'):
                self._load_jsonl(jsonl_file)

        print(f"Loaded {len(self.samples)} samples")

    def _load_jsonl(self, file_path: Path):
        """Load samples from a single JSONL file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    # Validate sample has required fields
                    if 'word' in sample and 'stroke3' in sample:
                        strokes = sample['stroke3']

                        # Skip empty or too short samples
                        if len(strokes) < 3:
                            continue

                        # Skip samples that are too long
                        if len(strokes) > self.max_stroke_len:
                            strokes = strokes[:self.max_stroke_len]
                            sample['stroke3'] = strokes

                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Get text
        word = sample['word']
        text_ids = self.vocab.encode(word)
        text_ids = text_ids[:self.max_text_len]

        # Get strokes
        strokes = np.array(sample['stroke3'], dtype=np.float32)

        # Normalize
        if self.normalize:
            strokes[:, :2] /= self.scale_factor

        # Augment
        if self.augment:
            strokes = augment_strokes(strokes)

        return {
            'word': word,
            'text_ids': torch.tensor(text_ids, dtype=torch.long),
            'strokes': torch.tensor(strokes, dtype=torch.float32),
            'text_len': len(text_ids),
            'stroke_len': len(strokes)
        }


def collate_fn(batch):
    """
    Collate function for DataLoader.
    Pads sequences to same length within batch.
    """
    words = [item['word'] for item in batch]
    text_ids = [item['text_ids'] for item in batch]
    strokes = [item['strokes'] for item in batch]
    text_lens = torch.tensor([item['text_len'] for item in batch])
    stroke_lens = torch.tensor([item['stroke_len'] for item in batch])

    # Pad sequences
    text_ids_padded = pad_sequence(text_ids, batch_first=True, padding_value=0)
    strokes_padded = pad_sequence(strokes, batch_first=True, padding_value=0)

    # Create mask for strokes
    max_stroke_len = strokes_padded.size(1)
    stroke_mask = torch.arange(max_stroke_len).unsqueeze(0) < stroke_lens.unsqueeze(1)

    return {
        'words': words,
        'text_ids': text_ids_padded,
        'text_lengths': text_lens,
        'strokes': strokes_padded,
        'stroke_lengths': stroke_lens,
        'stroke_mask': stroke_mask.float()
    }


def create_dataloader(
    data_path: str,
    vocab: MalayalamVocab,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **dataset_kwargs
) -> DataLoader:
    """Create DataLoader for training/evaluation."""
    dataset = MalayalamHandwritingDataset(data_path, vocab, **dataset_kwargs)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )


def split_dataset(
    data_path: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List, List, List]:
    """
    Split data into train/val/test sets.
    Returns lists of sample indices.
    """
    # Load all samples to get count
    samples = []
    path = Path(data_path)

    if path.is_file():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
    elif path.is_dir():
        for jsonl_file in path.glob('*.jsonl'):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))

    # Shuffle and split
    np.random.seed(seed)
    indices = np.random.permutation(len(samples))

    n_train = int(len(indices) * train_ratio)
    n_val = int(len(indices) * val_ratio)

    train_indices = indices[:n_train].tolist()
    val_indices = indices[n_train:n_train + n_val].tolist()
    test_indices = indices[n_train + n_val:].tolist()

    return train_indices, val_indices, test_indices
