"""
High-Performance Memory-Mapped DataLoader for YABGU Pretraining
Numpy memmap арқылы деректерді тікелей дигіден оқиды, бұл Google Colab RAM-ын бос ұстайды.
"""

import os
import numpy as np
import torch

class MemmapDataset:
    def __init__(self, data_path: str, seq_len: int = 1024):
        self.data_path = data_path
        self.seq_len = seq_len
        
        assert os.path.exists(data_path), f"Бинарлық дерек файлы табылмады: {data_path}"
        
        # Memory-mapped uint16 массив
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.num_tokens = len(self.data)
        self.num_samples = (self.num_tokens - 1) // self.seq_len
        
        assert self.num_tokens > self.seq_len, f"Дерек көлемі ({self.num_tokens}) seq_len ({self.seq_len})-ден үлкен болуы керек"

    def get_batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        # Кездейсоқ индекстер таңдау
        max_idx = self.num_tokens - self.seq_len - 1
        ix = torch.randint(0, max_idx, (batch_size,))
        
        x = torch.stack([torch.from_numpy((self.data[i : i + self.seq_len]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.seq_len]).astype(np.int64)) for i in ix])
        
        if "cuda" in device:
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
            
        return x, y
