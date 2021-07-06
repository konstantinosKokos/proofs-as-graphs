import torch

from ..types import Tensor, Module, List, Tuple

from torch_geometric.nn import MessagePassing
from torch_geometric.typing import PairTensor
from torch_geometric.utils import softmax as sparse_softmax

from torch.nn import Sequential, Dropout, Linear, ModuleList, LayerNorm, GELU
from torch import cat

from math import sqrt


class MultihopTransConv(Module):
    def __init__(self, layer_dims: List[int], edge_dim: int, residuals: List[Tuple[int, int]],
                 dropout_rate: float = 0.15, num_heads: int = 1,):
        super(MultihopTransConv, self).__init__()
        self.layers = ModuleList([BidirTransConv(input_dim, output_dim, edge_dim, dropout_rate, num_heads)
                                  for input_dim, output_dim in zip(layer_dims, layer_dims[1:])])
        self.residuals = dict(residuals)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        layer_outputs = [x]
        for i, layer in enumerate(self.layers):
            out = layer(layer_outputs[-1], edge_index, edge_attr)
            if i in self.residuals:
                out = out + layer_outputs[self.residuals[i] + 1]
            layer_outputs.append(out)
        return layer_outputs[-1]


class BidirTransConv(Module):
    def __init__(self, input_dim: int, output_dim: int, edge_dim: int, dropout_rate: float = 0.5, num_heads: int = 1):
        super(BidirTransConv, self).__init__()
        self.fw = TransConv(input_dim, output_dim // 2, edge_dim, dropout_rate, num_heads)
        self.bw = TransConv(input_dim, output_dim // 2, edge_dim, dropout_rate, num_heads)
        self.merge = Sequential(Linear(output_dim, 2 * output_dim), GELU(), Linear(2 * output_dim, output_dim))
        self.ln = LayerNorm(output_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        ret = cat((self.fw(x, edge_index, edge_attr), self.bw(x, edge_index.flip(0), edge_attr)), dim=-1)
        return self.ln(self.merge(ret) + ret)


class TransConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, edge_dim: int, dropout_rate: float = 0.5, num_heads: int = 1):
        super(TransConv, self).__init__(aggr='add', node_dim=0)
        self.dropout = Dropout(dropout_rate)
        self.q_projection = Linear(input_dim, output_dim)
        self.k_projection = Linear(input_dim + edge_dim, output_dim)
        self.v_projection = Linear(input_dim, output_dim)
        self.self_loop = Linear(input_dim, output_dim)
        self.gating = Linear(3 * output_dim, 1)
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.atn_dim = output_dim // self.num_heads

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x: PairTensor = (self.dropout(x), self.dropout(x))
        # propagate_type: (x: PairTensor, edge_attr: Tensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None).view(-1, self.output_dim)
        res = self.self_loop(x[0])
        gates = self.gating(cat((out, res, out - res), dim=-1)).sigmoid()
        return gates * out + (1 - gates) * res

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: Tensor, size_i: int) -> Tensor:
        qs = self.q_projection(x_i).view(-1, self.num_heads, self.atn_dim)
        ks = self.k_projection(cat((x_j, edge_attr), dim=-1)).view(-1, self.num_heads, self.atn_dim)
        vs = self.v_projection(x_j).view(-1, self.num_heads, self.atn_dim)
        atn = (qs * ks).sum(dim=-1) / sqrt(self.atn_dim)
        atn = sparse_softmax(atn, index, ptr, size_i)
        return atn.view(-1, self.num_heads, 1) * vs
