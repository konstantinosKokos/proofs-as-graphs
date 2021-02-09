from torch import Tensor
from torch_geometric.data import Data, Batch, DataLoader
from torch.nn import Module
from torch.optim import Optimizer

from typing import (List, Tuple, Union, Protocol, Dict, Any, TypeVar, Callable, Protocol, Sequence, overload,
                    Iterator, Set)
from typing import Optional as Maybe
from numpy import array


Metric = Dict[str, List[List]]
Epoch = Dict[str, List]


class OptLike(Protocol):
    def step(self) -> None:
        ...

    def zero_grad(self) -> None:
        ...


class Logger(Protocol):
    stats: Dict

    def log(self, *tensors: Tensor, train: bool) -> None:
        ...

    def register_epoch(self, train: bool, callback: Callable[['Logger'], None]) -> None:
        ...
