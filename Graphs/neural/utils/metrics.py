import torch
from ...types import Tensor, Tuple, List, Sequence
from torch.nn.functional import one_hot


@torch.no_grad()
def accuracy(predictions: Tensor, truths: Tensor) -> Tuple[int, int]:
    return predictions.shape[0], torch.sum(predictions.argmax(dim=-1).eq(truths)).item()


@torch.no_grad()
def per_class_accuracy(predictions: Tensor, truths: Tensor, num_classes: int = 3) -> Tuple[Tensor, Tensor]:
    xs = one_hot(predictions.argmax(dim=1), num_classes)
    ys = one_hot(truths, num_classes)
    return (xs.eq(ys) * ys).sum(dim=0), ys.sum(dim=0)


@torch.no_grad()
def epoch_accuracies(per_class_accs: List[Tuple[Tensor, Tensor]]):
    correct, total = list(map(sum, zip(*per_class_accs)))
    return (correct/total).tolist() + [(correct.sum()/total.sum()).item()]


def mean(xs: Sequence):
    return sum(xs) / len(xs)
