import torch
from ...typing import Tensor, Tuple, Sequence


@torch.no_grad()
def accuracy(predictions: Tensor, truths: Tensor) -> Tuple[int, int]:
    return predictions.shape[0], torch.sum(predictions.argmax(dim=-1).eq(truths)).item()


def mean(xs: Sequence):
    return sum(xs) / len(xs)
