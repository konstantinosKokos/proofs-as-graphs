from torch import Tensor
from torch_geometric.data import Data, Batch, DataLoader
from torch_geometric.typing import Adj, PairTensor
from torch.nn import Module
from torch.optim import Optimizer

from typing import (List, Tuple, Union, Protocol, Dict, Any, TypeVar, Callable, Protocol, Sequence, overload,
                    Iterator, Set, NamedTuple, Generic)
from typing import Optional as Maybe
from numpy import array

from dataclasses import dataclass

Metric = Dict[str, List[List]]
Epoch = Dict[str, List]


class OptLike(Protocol):
    lr: float

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


Node_co = TypeVar('Node_co', bound='Node', covariant=True)
T0 = TypeVar('T0')
T1 = TypeVar('T1')


@dataclass
class Node:
    index: int
    label: str

    def __hash__(self):
        return self.index


@dataclass
class ANode(Node):
    polarity: bool
    j_idx: int

    def __hash__(self):
        return self.index


@dataclass
class CNode(Node):

    def __hash__(self):
        return self.index


@dataclass
class WNode(Node):
    def __hash__(self):
        return self.index


class Edge(NamedTuple, Generic[T0]):
    target: Node
    label: T0


Graph = Dict[Node, Set[Edge]]


class GraphData(NamedTuple, Generic[T0, T1]):
    nodes: List[T0]                            # A list of node tokens
    edge_index: Tuple[List[int], List[int]]    # Edge_index in COO format
    edge_attrs: List[T0]                       # A list of edge labels
    roots: List[int]                           # A list of pointers to type anchors
    conclusion: int                            # A pointer to the conclusion atom
    words: List[T1]                            # A list of word tokens
