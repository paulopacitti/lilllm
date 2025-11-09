import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from lilllm.tokenizer import Tokenizer


class PretrainingDataset(Dataset):
    def __init__(
        self,
        text: str,
        tokenizer: Tokenizer,
        max_length: int,
        stride: int,
    ):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor]:
        return self.input_ids[i], self.target_ids[i]


def build_pretraining_dataloader(
    text: str,
    batch_size: int = 4,
    max_length: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    tokenizer = Tokenizer()
    dataset = PretrainingDataset(text, tokenizer, max_length, stride)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )
