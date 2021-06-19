from typing import TypeVar, Dict, Set, List, Tuple, Iterator, Union, NamedTuple, Generic, Callable, Sequence
from typing import Optional as Maybe

from dataclasses import dataclass

from torch import Tensor
from torch.nn import Module

from torch_geometric.data import Data, Batch

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
    link_idx: int

    def __hash__(self):
        return super(ANode, self).__hash__()


@dataclass
class TNode(Node):
    def __init__(self, index: int):
        super(TNode, self).__init__(index, label='(*)')
        
    def __hash__(self):
        return super(TNode, self).__hash__()


@dataclass
class CNode(Node):
    def __init__(self, index: int):
        super(CNode, self).__init__(index, label='(+)')

    def __hash__(self):
        return super(CNode, self).__hash__()


@dataclass
class WNode(Node):
    def __hash__(self):
        return super(WNode, self).__hash__()


Node_co = TypeVar('Node_co', bound='Node', covariant=True)


class Edge(NamedTuple, Generic[T0]):
    target: Node
    label: T0


Graph = Dict[Node, Set[Edge]]                   # Simple structure used to build up graphs


class GraphData(NamedTuple, Generic[T0, T1]):
    """
        Generic class to contain graphs with extra attributes.
    """
    nodes: List[T0]                             # A list of node tokens
    edge_index: Tuple[List[int], List[int]]     # Edge_index in COO format
    edge_attrs: List[T0]                        # A list of edge labels
    roots: List[int]                            # A list of pointers to type anchors
    conclusion: int                             # A pointer to the conclusion atom
    words: List[T1]                             # A list of word tokens


class ProofPair(Data):
    """
        Class representing two tensorized graphs intented for torch_geometric interfacing.
    """
    def __init__(self, p_graph: Tuple[Tensor, ...], h_graph: Tuple[Tensor, ...], y: Tensor):
        super(ProofPair, self).__init__()
        self.x_p, self.edge_index_p, self.edge_attr_p, self.word_pos_p, self.word_ids_p, self.word_starts_p = p_graph
        self.x_h, self.edge_index_h, self.edge_attr_h, self.word_pos_h, self.word_ids_h, self.word_starts_h = h_graph
        self.y = y

    def __inc__(self, key, value):
        return (self.x_h.shape[0] if key in {'edge_index_h', 'word_pos_h', 'conc_h'} else
                self.x_p.shape[0] if key in {'edge_index_p', 'word_pos_p', 'conc_p'} else
                super(ProofPair, self).__inc__(key, value))
