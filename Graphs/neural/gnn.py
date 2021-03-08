from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as sparse_softmax
from torch.nn import Linear, Sequential, GELU, LayerNorm, Dropout, ModuleList
from torch import cat, dropout
from ..typing import Tensor, Module
from torch_geometric.typing import Adj, SparseTensor, OptPairTensor, PairTensor
from torch_sparse import matmul
from math import sqrt


class BGraphConvWrapper(Module):
    def __init__(self, dim: int, num_layers: int):
        super(BGraphConvWrapper, self).__init__()
        self.fws = ModuleList([BGraphConv(dim) for _ in range(num_layers)])
        self.bws = ModuleList([BGraphConv(dim) for _ in range(num_layers)])
        self.merge = ModuleList([Sequential(Linear(3 * dim, dim, False), GELU()) for _ in range(num_layers)])
        self.lns = ModuleList([LayerNorm(dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        for fw, bw, m, ln in zip(self.fws, self.bws, self.merge, self.lns):
            fw_x = fw(x, edge_index, edge_attr)
            bw_x = bw(x, edge_index.flip(0), edge_attr)
            x += m(cat((x, fw_x, bw_x), dim=-1))
            x = ln(x)
        return x


class BGraphConv(MessagePassing):
    def __init__(self, dim: int):
        super(BGraphConv, self).__init__(aggr='mean')
        self.mlp_n = Sequential(Linear(dim, dim), Dropout(0.5))
        self.mlp_e = Sequential(Linear(dim, dim), Dropout(0.5))
        self.merge = Sequential(Linear(3 * dim, dim), Dropout(0.5), GELU())

    def reset_parameters(self) -> None:
        self.mlp_n.reset_parameters()
        self.mlp_e.reset_parameters()

    def forward(self, x: Tensor, edge_index: Adj, edge_attr: Tensor) -> Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.merge(cat((x_i, self.mlp_e(edge_attr), self.mlp_n(x_j)), dim=-1))


class TConvWrapper(Module):
    def __init__(self, dim: int, edge_dim: int, num_layers: int, num_heads: int = 1, weight_sharing: bool = False):
        super(TConvWrapper, self).__init__()
        if weight_sharing:
            self.conv = BiTransConv(dim, edge_dim, num_heads, 0.15)
        else:
            self.convs = ModuleList([BiTransConv(dim, edge_dim, num_heads, 0.15) for _ in range(num_layers)])
        self.weight_sharing = weight_sharing
        self.num_layers = num_layers

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        for layer in range(self.num_layers):
            conv = self.conv if self.weight_sharing else self.convs[layer]
            x = conv(x, edge_index, edge_attr)
        return x


class BiTransConv(Module):
    def __init__(self, dim: int, edge_dim: int, num_heads: int, dropout_rate: float):
        super(BiTransConv, self).__init__()
        self.fw = TransConv(dim, dim, edge_dim, dropout_rate, num_heads)
        self.bw = TransConv(dim, dim, edge_dim, dropout_rate, num_heads)
        self.merge = Sequential(Dropout(dropout_rate), Linear(3 * dim, dim, False), GELU(), LayerNorm(dim))

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.merge(cat((x,
                               self.fw(x, edge_index, edge_attr),
                               self.bw(x, edge_index.flip(0), edge_attr)),
                              dim=-1))


class TransConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, edge_dim: int, dropout_rate: float, num_heads: int = 1):
        super(TransConv, self).__init__(aggr='add', node_dim=0)
        self.q_projection = Sequential(Dropout(dropout_rate), Linear(input_dim, output_dim))
        self.k_projection = Sequential(Dropout(dropout_rate), Linear(input_dim + edge_dim, output_dim))
        self.v_projection = Sequential(Dropout(dropout_rate), Linear(input_dim, output_dim))
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_dim = edge_dim
        self.atn_dim = output_dim // num_heads

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: Tensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out.view(-1, self.output_dim)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: Tensor, size_i: int) -> Tensor:
        queries = self.q_projection(x_i).view(-1, self.num_heads, self.atn_dim)
        keys = self.k_projection(cat((x_j, edge_attr), dim=-1)).view(-1, self.num_heads, self.atn_dim)
        values = self.v_projection(x_j).view(-1, self.num_heads, self.atn_dim)
        atn = (queries * keys).sum(dim=-1) / sqrt(self.atn_dim)
        atn = sparse_softmax(atn, index, ptr, size_i)
        return atn.view(-1, self.num_heads, 1) * values
