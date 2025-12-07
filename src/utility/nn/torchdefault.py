from os import PathLike
from typing import IO
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset

def set_defaults():
    """
    A
    """
    torch.set_default_device('cpu')
    torch.set_default_dtype(torch.double)
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
        torch.set_default_dtype(torch.double)

def tensor(t, **kwargs):
    if isinstance(t, torch.Tensor):
        return t.to(
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
            **kwargs
        )
    return torch.tensor(
        data=t, dtype=torch.get_default_dtype(),
        device=torch.get_default_device(), **kwargs
    )

def tensor_dataset(*tensors, **kwargs):
    return TensorDataset(*[ tensor(t, **kwargs) for t in tensors ])

def generator(seed: int | None = None):
    gen = torch.Generator(device=torch.get_default_device())
    if seed is not None:
        gen.manual_seed(seed)
    return gen 

def save(state_dict: dict, f: str | PathLike[str] | IO[bytes], **kwargs):
    torch.save(state_dict, f, **kwargs)
    
def load(f: str | PathLike[str] | IO[bytes], **kwargs):
    return torch.load(f, map_location=torch.get_default_device(), weights_only=True, **kwargs)

def sequential_dataloader(dataset: Dataset, **kwargs):
    return DataLoader(dataset=dataset, batch_size=None, **kwargs)

def batch_dataloader(
    dataset: Dataset,     batch_size: int | None = None,
    shuffle: bool = True, drop_last: bool = True, seed: int | None = None,
    **kwargs
):
    return DataLoader(
        dataset=dataset,     batch_size=batch_size, shuffle=shuffle,
        drop_last=drop_last, generator=generator(seed),
        **kwargs
    )