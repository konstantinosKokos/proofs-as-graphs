from torch_geometric.utils import to_dense_batch, softmax
from ..typing import Tensor, Tuple
from ..data.tokenizer import Tokenizer
from ..neural.embedding import InvertibleEmbedder
from ..neural.encoders import Encoder
from ..neural.gnn import TConvWrapper
from ..neural.mha import MultiHeadAttention

from torch.nn import Module, Linear, Sequential, Dropout, Embedding, Tanh, LSTM, GELU, LayerNorm
from torch.nn.functional import pad
from torch import cat, save, bmm


class Base(Module):
    def __init__(self, tokenizer: Tokenizer, num_layers: int, device: str = 'cuda'):
        super(Base, self).__init__()
        self.ndim = 128
        self.tokenizer = tokenizer
        self.atom_embedding = InvertibleEmbedder(len(tokenizer.atom_map), self.ndim).to(device)
        self.edge_embedding = Embedding(3, self.ndim).to(device)
        self.gnn = TConvWrapper(self.ndim, self.ndim, 0.5, num_layers, 2).to(device)
        # self.gnn = BGraphConv(self.ndim, 'mean').to(device)
        self.num_layers = num_layers
        self.device = device
        self.dropout = Dropout(0.5)

    def embed_atoms(self, atoms: Tensor) -> Tensor:
        return self.atom_embedding.embed(atoms).squeeze(1)

    def embed_nodes(self, atoms: Tensor) -> Tensor:
        return self.embed_atoms(atoms)

    def embed_edges(self, edges: Tensor) -> Tensor:
        return self.edge_embedding(edges)

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def contextualize_nodes(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor) -> Tensor:
        vectors = self.dropout(self.embed_nodes(atoms))
        edge_attr = self.dropout(self.embed_edges(edge_ids))
        return self.gnn(vectors, edge_index, edge_attr)

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base):
        super(PairClassifier, self).__init__()
        self.base = base
        self.word_encoder = Encoder(base.tokenizer, base.device, hidden_size=128)
        self.decoder = LSTM(input_size=128, hidden_size=64, batch_first=True, bidirectional=True).to(base.device)
        self.collapser = Sequential(Linear(2 * 256, 128, True), GELU(), Dropout(0.5), LayerNorm(128)).to(base.device)
        self.combiner = Sequential(Linear(128 * 4, 64), Dropout(0.5), Tanh(), Linear(64, 3)).to(base.device)
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
        mask_h = n_h.eq(0).all(dim=-1)                                                                      # B, S1
        mask_p = n_p.eq(0).all(dim=-1)                                                                      # B, S2
        mask = mask_h.unsqueeze(-1).repeat(1, 1, nwp).bitwise_or(mask_p.unsqueeze(1).repeat(1, nwh, 1))     # B, S1, S2
        cross_atn = bmm(n_h, n_p.permute(0, 2, 1))                                                          # B, S1, S2
        cross_atn[mask] = -1e10
        ctx_hp = bmm(cross_atn.softmax(dim=1).permute(0, 2, 1), ctx_h)                                      # B, S2, d
        ctx_ph = bmm(cross_atn.softmax(dim=2), ctx_p)                                                       # B, S1, d
        ctx_h = self.collapser(cat((ctx_h - ctx_ph, ctx_h * ctx_ph), dim=-1))                               # B, S1, d'
        ctx_p = self.collapser(cat((ctx_p - ctx_hp, ctx_p * ctx_hp), dim=-1))                               # B, S2, d'
        ctx_h = pad(ctx_h, (0, 0, 0, max(0, nwp - nwh)), 'constant', 0.)                                    # B, S, d'
        mask_h = pad(mask_h.t(), (0, 0, 0, max(0, nwp - nwh)), 'constant', True).t()                        # B, S
        ctx_p = pad(ctx_p, (0, 0, 0, max(0, nwh - nwp)), 'constant', 0.)                                    # B, S, d'
        mask_p = pad(mask_p.t(), (0, 0, 0, max(0, nwh - nwp)), 'constant', True).t()                        # B, S
        ctx_h = ctx_h.masked_fill(mask_h.unsqueeze(-1), 0)
        ctx_p = ctx_p.masked_fill(mask_p.unsqueeze(-1), 0)
        d_out, _ = self.decoder(cat((ctx_h, ctx_p), dim=0))                                                 # 2B, S, d''
        ctx_h, ctx_p = d_out.chunk(2, dim=0)
        ctx_h = self.dropout(ctx_h.masked_fill(mask_h.unsqueeze(-1), 0))                                    # B, S, d''
        ctx_p = self.dropout(ctx_p.masked_fill(mask_p.unsqueeze(-1), 0))                                    # B, S, d''
        ctx_h = cat((ctx_h.max(dim=1)[0], ctx_h.sum(dim=1)/mask_h.logical_not().sum(dim=1).unsqueeze(-1)),  # B, S, 2d''
                    dim=-1)
        ctx_p = cat((ctx_p.max(dim=1)[0], ctx_p.sum(dim=1)/mask_p.logical_not().sum(dim=1).unsqueeze(-1)),  # B, S, 2d''
                    dim=-1)
        return self.combiner(cat((ctx_h, ctx_p), dim=-1)), cross_atn

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)
