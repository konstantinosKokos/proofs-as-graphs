from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch, softmax
from torch.nn.functional import dropout
from ..typing import Tensor, Tuple, PairTensor
from ..data.tokenizer import Tokenizer
from ..neural.embedding import InvertibleEmbedder
from ..neural.encoders import Encoder
from ..neural.mha import MultiHeadAttention
from math import sqrt

from torch.nn import Module, ModuleList, Linear, Sequential, Dropout, GELU, Embedding, Tanh, LayerNorm, LSTM
from torch import cat, save, bmm, zeros_like


class TransConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, edge_attrs: int, dropout_rate: float, num_heads: int = 1):
        super(TransConv, self).__init__(aggr='add', node_dim=0)
        self.q_projection = Linear(input_dim, output_dim, False)
        self.k_projection = Linear(input_dim, output_dim, False)
        self.v_projection = Linear(input_dim, output_dim, False)
        self.e_projection = Sequential(Linear(edge_attrs, output_dim), Tanh())
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_attrs = edge_attrs
        self.dropout = dropout_rate

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: Tensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out.view(-1, self.num_heads * self.output_dim)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: Tensor, size_i: int) -> Tensor:
        queries = dropout(self.q_projection(x_i).view(-1, self.num_heads, self.output_dim), self.dropout, self.training)
        keys = dropout(self.k_projection(x_j).view(-1, self.num_heads, self.output_dim), self.dropout, self.training)
        values = dropout(self.v_projection(x_j).view(-1, self.num_heads, self.output_dim), self.dropout, self.training)
        edges = dropout(self.e_projection(edge_attr).view(-1, self.num_heads, self.output_dim), self.dropout, self.training)
        atn = softmax((queries*(keys * edges)).sum(dim=-1) / sqrt(self.output_dim), index, ptr, size_i)
        return atn.view(-1, self.num_heads, 1) * values


class Base(Module):
    def __init__(self, tokenizer: Tokenizer, num_layers: int, device: str = 'cuda'):
        super(Base, self).__init__()
        self.ndim = 64
        self.tokenizer = tokenizer
        self.atom_embedding = InvertibleEmbedder(len(tokenizer.atom_map), self.ndim).to(device)
        self.edge_embedding = Embedding(3, self.ndim).to(device)
        self.c2r = ModuleList([TransConv(self.ndim, self.ndim, self.ndim, 0.5) for _ in range(num_layers)]).to(device)
        self.r2c = ModuleList([TransConv(self.ndim, self.ndim, self.ndim, 0.5) for _ in range(num_layers)]).to(device)
        self.ffn = ModuleList([Sequential(Linear(2*self.ndim, self.ndim, False), Dropout(0.5))
                               for _ in range(num_layers)]).to(device)
        self.lns = ModuleList([LayerNorm(self.ndim) for _ in range(num_layers)]).to(device)
        self.num_layers = num_layers
        self.device = device
        self.dropout = Dropout(0.5)

    def embed_atoms(self, atoms: Tensor) -> Tensor:
        return self.atom_embedding.embed(atoms).squeeze(1)

    def embed_nodes(self, atoms: Tensor) -> Tensor:
        return self.embed_atoms(atoms)
        # words, ids = to_dense_batch(words, word_batch)
        # w_vectors = self.word_encoder.embed_words(words)[ids][word_starts.eq(1)]
        # atom_vectors.index_put_((word_pos,), self.word_encoder.w2g(w_vectors))
        # return atom_vectors

    def embed_edges(self, edges: Tensor) -> Tensor:
        return self.edge_embedding(edges)

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def contextualize_nodes(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor) -> Tensor:
        vectors = self.dropout(self.embed_nodes(atoms))
        edge_attr = self.dropout(self.embed_edges(edge_ids))
        for hop in range(self.num_layers):
            r2c = self.r2c[hop](x=vectors, edge_index=edge_index, edge_attr=edge_attr)
            c2r = self.c2r[hop](x=vectors, edge_index=edge_index.flip(0), edge_attr=edge_attr)
            vectors = self.lns[hop](self.ffn[hop](cat((r2c, c2r), dim=-1)) + vectors)
        return vectors

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base):
        super(PairClassifier, self).__init__()
        self.base = base
        self.word_encoder = Encoder(base.tokenizer, base.device)
        self.collapser = Sequential(Linear(3 * 512, 256), GELU(), Dropout(0.5)).to(base.device)
        self.combiner = Sequential(Linear(256 * 4, 256), GELU(), Dropout(0.5), Linear(256, 3)).to(base.device)
        self.mha = MultiHeadAttention(4, self.base.ndim, self.base.ndim, self.base.ndim//4).to(base.device)
        self.h_encoder = LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True).to(base.device)
        self.p_encoder = LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True).to(base.device)
        self.dropout = Dropout(0.5)

    def readout(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor, word_pos: Tensor, word_batch: Tensor,
                word_ids: Tensor, word_starts: Tensor) -> Tuple[Tensor, Tensor]:
        node_reprs = self.base.contextualize_nodes(atoms, edge_index, edge_ids)[word_pos]
        words, ids = to_dense_batch(word_ids, word_batch, fill_value=self.word_encoder.pad_value)
        ctx = self.dropout(self.word_encoder(words)[ids][word_starts.eq(1)])
        ctx, _ = to_dense_batch(ctx, word_batch[word_starts.eq(1)])
        node_reprs, _ = to_dense_batch(node_reprs, word_batch[word_starts.eq(1)], fill_value=0)
        return ctx, node_reprs

    def entail(self, ctx_h: Tensor, n_h: Tensor, ctx_p: Tensor, n_p: Tensor) -> Tuple[Tensor, Tensor]:
        nwh, nwp = n_h.shape[1], n_p.shape[1]
        mask_h = n_h.eq(0).all(dim=-1)                                              # B, S1
        mask_p = n_p.eq(0).all(dim=-1)                                              # B, S2
        atn_ph = self.mha(n_h, n_p, mask_h)
        # atn_hp = self.mha(n_p, n_h, mask_p)
        ctx_ph = bmm(atn_ph, ctx_p)
        # ctx_hp = bmm(atn_hp, ctx_h)
        ctx_h = self.collapser(self.dropout(cat((ctx_h, ctx_h - ctx_ph, ctx_h * ctx_ph), dim=-1)))
        # ctx_p = self.collapser(self.dropout(cat((ctx_p, ctx_p - ctx_hp, ctx_p * ctx_hp), dim=-1)))
        ctx_h, _ = self.h_encoder(ctx_h)
        # ctx_p, _ = self.p_encoder(ctx_p)
        ctx_h = self.dropout(ctx_h.masked_fill(mask_h.unsqueeze(-1), 0))
        # ctx_p = self.dropout(ctx_p.masked_fill(mask_p.unsqueeze(-1), 0))
        # ctx_h = cat((ctx_h.max(dim=1)[0], ctx_h.sum(dim=1)/mask_h.logical_not().sum(dim=1).unsqueeze(-1)), dim=-1)
        # ctx_p = cat((ctx_p.max(dim=1)[0], ctx_p.sum(dim=1)/mask_p.logical_not().sum(dim=1).unsqueeze(-1)), dim=-1)
        # return self.combiner(cat((ctx_h, ctx_p), dim=-1))

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)

