from torch_geometric.nn import TransformerConv, MessagePassing, global_mean_pool, global_max_pool
from torch_geometric.utils import to_dense_batch, softmax
from ..typing import Tensor, Dict, Tuple, array
from ..neural.embedding import InvertibleEmbedder
from ..neural.encoders.lstm import Vanilla
from torch.nn import Module, ModuleList, Linear, Sequential, Dropout, GELU, Embedding
from torch import cat, save, bmm


class Base(Module):
    def __init__(self, atom_map: Dict[str, int], num_layers: int, device: str = 'cuda'):
        super(Base, self).__init__()
        ndim = 64
        edim = 12
        self.atom_embedding = InvertibleEmbedder(len(atom_map), ndim).to(device)
        self.edge_embedding = Embedding(3, edim).to(device)
        self.r2c = ModuleList([TransformerConv(ndim, ndim, edge_dim=edim) for _ in range(num_layers)]).to(device)
        self.c2r = ModuleList([TransformerConv(ndim, ndim, edge_dim=edim) for _ in range(num_layers)]).to(device)
        self.aggr = Sequential(Linear(ndim * 2, ndim), GELU()).to(device)
        # self.word_encoder = Vanilla(300, 256, gdim, embedding_table).to(device)
        self.num_layers = num_layers
        self.atom_map = atom_map
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

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def contextualize_nodes(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor) -> Tensor:
        vectors = self.dropout(self.embed_nodes(atoms))
        edge_attr = self.edge_embedding(edge_ids)
        for hop in range(self.num_layers):
            r2c = self.r2c[hop](x=vectors, edge_index=edge_index, edge_attr=edge_attr)
            c2r = self.c2r[hop](x=vectors, edge_index=edge_index.flip(0), edge_attr=edge_attr)
            vectors = self.aggr(self.dropout(cat((r2c, c2r), dim=-1)))
        return vectors

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base, table: array):
        super(PairClassifier, self).__init__()
        self.base = base
        self.word_encoder = Vanilla(word_dim=300, hidden_dim=256, graph_dim=64, embedding_table=table).to(base.device)
        self.expander = Sequential(Linear(64, 256), GELU(), Dropout(0.5), Linear(256, 512)).to(base.device)
        self.collapser = Sequential(Linear(3 * 512, 256), GELU(), Dropout(0.5)).to(base.device)
        self.combiner = Sequential(Linear(256 * 4, 256), GELU(), Dropout(0.5), Linear(256, 3)).to(base.device)

    def readout(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor, word_pos: Tensor, word_batch: Tensor,
                word_ids: Tensor, word_starts: Tensor) -> Tuple[Tensor, Tensor]:
        node_reprs = self.base.contextualize_nodes(atoms, edge_index, edge_ids)[word_pos]
        # atn = softmax(self.expander(node_reprs), word_batch[word_starts.eq(1)])
        words, ids = to_dense_batch(word_ids, word_batch)
        ctx = self.word_encoder(words)[ids][word_starts.eq(1)]
        ctx, _ = to_dense_batch(ctx, word_batch[word_starts.eq(1)])
        node_reprs, _ = to_dense_batch(node_reprs, word_batch[word_starts.eq(1)])
        return ctx, node_reprs

    def entail(self, ctx_h: Tensor, n_h: Tensor, ctx_p: Tensor, n_p: Tensor):
        cross_atn = bmm(n_h, n_p.permute(0, 2, 1))
        ctx_hp = bmm(cross_atn.softmax(dim=1).permute(0, 2, 1), ctx_h)
        ctx_ph = bmm(cross_atn.softmax(dim=2), ctx_p)
        ctx_h = self.collapser(cat((ctx_h, ctx_h - ctx_ph, ctx_h * ctx_ph), dim=-1))
        ctx_p = self.collapser(cat((ctx_p, ctx_p - ctx_hp, ctx_p * ctx_hp), dim=-1))
        ctx_h = cat((ctx_h.max(dim=1)[0], ctx_h.mean(dim=1)), -1)
        ctx_p = cat((ctx_p.max(dim=1)[0], ctx_p.mean(dim=1)), -1)
        return self.combiner(cat((ctx_h, ctx_p), dim=-1))

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


# class PairClassifier(Module):
#     def __init__(self, base: Base, device: str):
#         super(PairClassifier, self).__init__()
#         dim = 512
#         self.base = base
#         self.combiner = Sequential(Linear(dim*8, dim*2), GELU(), Dropout(0.5), Linear(dim*2, 3)).to(device)
#         self.projector = Sequential(Linear(32, 256, False), GELU(), Dropout(0.5), Linear(256, 512, False)).to(device)
#         self.device = device
#
#     def readout(self, atoms: Tensor, words: Tensor, w_pos: Tensor, w_batch: Tensor, w_starts: Tensor,
#                 concs: Tensor, edge_index: Tensor, batch: Tensor) -> Tensor:
#         atn_vectors = self.base.contextualize_nodes(atoms, words, w_pos, w_batch, w_starts, edge_index)
#         atn_vectors = softmax(self.projector(atn_vectors), w_batch)
#         ctx_vectors = self.base.embed_words(words, w_batch, w_starts)
#         out_vectors = atn_vectors * ctx_vectors
#         return cat((global_add_pool(out_vectors, w_batch), global_mean_pool(out_vectors, w_batch)), dim=-1)
#
#     def entail(self, x1: Tensor, x2: Tensor):
#         x = cat((x1, x2, x1 * x2, x1 - x2), dim=-1)
#         return self.combiner(x)
#
