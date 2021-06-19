from ..types import Module, Tensor, List, Maybe, Tuple, Batch
from .gnn import MultihopTransConv
from .tokenizer import BertWrapper

from math import sqrt

from torch import cat, cartesian_prod
from torch.nn import Embedding, LayerNorm, Linear, Sequential
from torch.nn.functional import pad
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import global_mean_pool


class GNN(Module):
    def __init__(self, layer_dims: List[int], edge_dim: int, num_nodes: int, num_edges: int):
        super(GNN, self).__init__()
        self.node_embedding = Embedding(num_nodes, layer_dims[0])
        self.edge_embedding = Embedding(num_edges, edge_dim)
        self.gnn = MultihopTransConv(layer_dims, edge_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, anchor_pos: Maybe[Tensor] = None):
        node_attr = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        ctx = self.gnn(node_attr, edge_index, edge_attr)
        return ctx if anchor_pos is None else ctx[anchor_pos]


class PairClassifier(Module):
    def __init__(self, layer_dims: List[int], edge_dim: int, num_nodes: int,
                 num_edges: int, bert: BertWrapper):
        super(PairClassifier, self).__init__()
        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.gnn = GNN(layer_dims, edge_dim, num_nodes, num_edges)
        self.bert = bert
        self.cls = Sequential(LayerNorm(768), Linear(768, 3))

    def readouts(self, x_p: Tensor, edge_index_p: Tensor, edge_attr_p: Tensor, word_pos_p: Tensor,
                 x_h: Tensor, edge_index_h: Tensor, edge_attr_h: Tensor, word_pos_h: Tensor) -> Tuple[Tensor, Tensor]:
        ctx_p = self.gnn(x_p, edge_index_p, edge_attr_p, word_pos_p)
        ctx_h = self.gnn(x_h, edge_index_h, edge_attr_h, word_pos_h)
        return ctx_p, ctx_h

    def bert_vectors(self, word_ids_p: Tensor, word_ids_p_batch: Tensor, word_starts_p: Tensor,
                     word_ids_h: Tensor, word_ids_h_batch: Tensor, word_starts_h: Tensor) -> Tuple[Tensor, Tensor]:
        p_ids, p_mask = to_dense_batch(word_ids_p, word_ids_p_batch, self.bert.tokenizer.pad_token_id)
        h_ids, h_mask = to_dense_batch(word_ids_h, word_ids_h_batch, self.bert.tokenizer.pad_token_id)
        hpad = (0, max(0, p_ids.shape[1] - h_ids.shape[1]))
        ppad = (0, max(0, h_ids.shape[1] - p_ids.shape[1]))
        h_ids = pad(h_ids, hpad, value=self.bert.tokenizer.pad_token_id)
        h_mask = pad(h_mask, hpad, value=False)
        p_ids = pad(p_ids, ppad, value=self.bert.tokenizer.pad_token_id)
        p_mask = pad(p_mask, ppad, value=False)
        ids = cat((p_ids, h_ids), dim=0)
        mask = cat((p_mask, h_mask), dim=0)
        lhs_p, lhs_h = self.bert.model.forward(ids, mask).last_hidden_state.chunk(2, dim=0)
        return lhs_p[p_mask][word_starts_p.eq(1)], lhs_h[h_mask][word_starts_h.eq(1)]

    def cross_attention(self, ctx_p: Tensor, ctx_h: Tensor, lhs_p: Tensor, lhs_h: Tensor,
                        batch_p: Tensor, batch_h: Tensor) -> Tensor:
        np, nh = batch_p.shape[0], batch_h.shape[0]
        mask_p, mask_h = cartesian_prod(batch_h, batch_p).view(nh, np, -1).chunk(2, dim=-1)
        mask_hp = mask_p.eq(mask_h).squeeze()
        atn = ctx_h @ ctx_p.t() / sqrt(self.output_dim)
        atn[mask_hp] = -1e08
        atn = atn.softmax(dim=-1)
        cls = self.cls(lhs_h + atn @ lhs_p)
        return global_mean_pool(cls.softmax(dim=-1), batch_h)

    def batch_to_label(self, batch: Batch) -> Tensor:
        word_reprs_p, word_reprs_h = self.bert_vectors(
            batch.word_ids_p, batch.word_ids_p_batch, batch.word_starts_p,
            batch.word_ids_h, batch.word_ids_h_batch, batch.word_starts_h)
        node_reprs_p, node_reprs_h = self.readouts(
            batch.x_p, batch.edge_index_p, batch.edge_attr_p, batch.word_pos_p,
            batch.x_h, batch.edge_index_h, batch.edge_attr_h, batch.word_pos_h)
        return self.cross_attention(node_reprs_p, node_reprs_h,
                                    word_reprs_p, word_reprs_h,
                                    batch.word_ids_p_batch[batch.word_starts_p.eq(1)],
                                    batch.word_ids_h_batch[batch.word_starts_h.eq(1)])