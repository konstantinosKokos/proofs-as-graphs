import torch
from ...typing import Tensor, Tuple, Sequence, List


@torch.no_grad()
def accuracy(predictions: Tensor, truths: Tensor) -> Tuple[int, int]:
    return predictions.shape[0], torch.sum(predictions.argmax(dim=-1).eq(truths)).item()


@torch.no_grad()
def per_class_accuracy(predictions: Tensor, truths: Tensor) -> Tuple[List[int], List[int]]:
    total = [truths.eq(c).sum().item() for c in range(predictions.shape[1])]
    correct = [torch.sum(predictions[truths.eq(c)].argmax(dim=-1).eq(truths[truths.eq(c)])).item()
               if total[c] != 0 else 0
               for c in range(predictions.shape[1])]
    return total, correct


def mean(xs: Sequence):
    return sum(xs) / len(xs)
