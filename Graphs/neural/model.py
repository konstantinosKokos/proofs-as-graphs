from torch_geometric.nn import TransformerConv, MessagePassing
from torch_geometric.utils import to_dense_batch, softmax
from torch.nn.functional import dropout
from ..typing import Tensor, Tuple, PairTensor
from ..data.tokenizer import Tokenizer
from ..neural.embedding import InvertibleEmbedder
from ..neural.encoders import Encoder
from math import sqrt

from torch.nn import Module, ModuleList, Linear, Sequential, Dropout, GELU, Embedding, GRUCell
from torch import cat, save, bmm


class TransConv(MessagePassing):
    def __init__(self, input_dim: int, output_dim: int, edge_types: int, dropout_rate: float, num_heads: int = 1):
        super(TransConv, self).__init__(aggr='add', node_dim=0)
        self.q_projection = Linear(input_dim, output_dim, False)
        self.k_projection = Linear(input_dim, output_dim, False)
        self.v_projection = Linear(input_dim, output_dim, False)
        self.e_projection = Embedding(edge_types, output_dim)
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_types = edge_types
        self.dropout = dropout_rate

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor):
        x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor, edge_attr: Tensor)
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr, size=None)
        out = out.view(-1, self.num_heads * self.output_dim)
        return out

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor, index: Tensor, ptr: Tensor, size_i: int) -> Tensor:
        queries = self.q_projection(dropout(x_i, self.dropout, self.training)).view(-1, self.num_heads, self.output_dim)
        keys = self.k_projection(dropout(x_j, self.dropout, self.training)).view(-1, self.num_heads, self.output_dim)
        values = self.v_projection(dropout(x_j, self.dropout, self.training)).view(-1, self.num_heads, self.output_dim)
        edges = self.e_projection(edge_attr).view(-1, self.num_heads, self.output_dim)
        atn = softmax((queries*(keys+edges)).sum(dim=-1) / sqrt(self.output_dim), index, ptr, size_i)
        return atn.view(-1, self.num_heads, 1) * values


class Base(Module):
    def __init__(self, num_atoms: int, num_layers: int, device: str = 'cuda'):
        super(Base, self).__init__()
        ndim = 64
        self.atom_embedding = InvertibleEmbedder(num_atoms, ndim).to(device)
        self.r2c = ModuleList([TransConv(ndim, ndim, 3, 0.15) for _ in range(num_layers)]).to(device)
        self.c2r = ModuleList([TransConv(ndim, ndim, 3, 0.15) for _ in range(num_layers)]).to(device)
        self.aggr = ModuleList([Sequential(Linear(ndim * 2, ndim), GELU()) for _ in range(num_layers)]).to(device)
        self.num_layers = num_layers
        self.device = device
        self.dropout = Dropout(0.15)

    def embed_atoms(self, atoms: Tensor) -> Tensor:
        return self.atom_embedding.embed(atoms).squeeze(1)

    def embed_nodes(self, atoms: Tensor) -> Tensor:
        return self.embed_atoms(atoms)
        # words, ids = to_dense_batch(words, word_batch)
        # w_vectors = self.word_encoder.embed_words(words)[ids][word_starts.eq(1)]
        # atom_vectors.index_put_((word_pos,), self.word_encoder.w2g(w_vectors))
        # return atom_vectors

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def contextualize_nodes(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor) -> Tensor:
        vectors = self.dropout(self.embed_nodes(atoms))
        for hop in range(self.num_layers):
            r2c = self.r2c[hop](x=vectors, edge_attr=edge_ids, edge_index=edge_index)
            c2r = self.c2r[hop](x=vectors, edge_attr=edge_ids, edge_index=edge_index.flip(0))
            # vectors = r2c + c2r + vectors
            vectors = self.aggr[hop](self.dropout(cat((r2c + vectors, c2r), dim=-1)))
        return vectors

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base, tokenizer: Tokenizer):
        super(PairClassifier, self).__init__()
        self.base = base
        self.tokenizer = tokenizer
        self.word_encoder = Encoder(tokenizer, base.device)
        self.collapser = Sequential(Linear(2 * 512, 256), GELU(), Dropout(0.5)).to(base.device)
        self.combiner = Sequential(Linear(256 * 4, 256), GELU(), Dropout(0.5), Linear(256, 3)).to(base.device)
        self.dropout = Dropout(0.5)

    def readout(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor, word_pos: Tensor, word_batch: Tensor,
                word_ids: Tensor, word_starts: Tensor) -> Tuple[Tensor, Tensor]:
        node_reprs = self.base.contextualize_nodes(atoms, edge_index, edge_ids)[word_pos]
        words, ids = to_dense_batch(word_ids, word_batch, fill_value=self.word_encoder.pad_value)
        ctx = self.dropout(self.word_encoder(words)[ids][word_starts.eq(1)])
        ctx, _ = to_dense_batch(ctx, word_batch[word_starts.eq(1)])
        node_reprs, _ = to_dense_batch(node_reprs, word_batch[word_starts.eq(1)])
        return ctx, node_reprs

    def entail(self, ctx_h: Tensor, n_h: Tensor, ctx_p: Tensor, n_p: Tensor) -> Tuple[Tensor, Tensor]:
        cross_atn = bmm(n_h, n_p.permute(0, 2, 1))
        mask_h = n_h.sum(dim=-1, keepdim=True).eq(0).repeat(1, 1, n_p.shape[1])
        mask_p = n_p.sum(dim=-1).eq(0).unsqueeze(1).repeat(1, n_h.shape[1], 1)
        mask = mask_h.bitwise_or(mask_p)
        cross_atn[mask] = -1e10
        ctx_hp = bmm(cross_atn.softmax(dim=1).permute(0, 2, 1), ctx_h)
        ctx_ph = bmm(cross_atn.softmax(dim=2), ctx_p)
        ctx_h = ctx_h + ctx_ph
        ctx_p = ctx_p + ctx_hp

        ctx_h = self.collapser(cat((ctx_h - ctx_ph, ctx_h * ctx_ph), dim=-1))
        ctx_p = self.collapser(cat((ctx_p - ctx_hp, ctx_p * ctx_hp), dim=-1))
        ctx_h = cat((ctx_h.max(dim=1)[0], ctx_h.mean(dim=1)), -1)
        ctx_p = cat((ctx_p.max(dim=1)[0], ctx_p.mean(dim=1)), -1)
        return self.combiner(cat((ctx_h, ctx_p), dim=-1)), cross_atn

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)

