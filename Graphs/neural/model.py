from torch_geometric.utils import to_dense_batch
from torch_geometric.utils import softmax as sparse_softmax
from torch_geometric.nn import global_max_pool
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
        self.enc_dim = 128
        self.tokenizer = tokenizer
        self.atom_embedding = InvertibleEmbedder(len(tokenizer.atom_map), self.ndim).to(device)
        self.edge_embedding = Embedding(3, self.ndim).to(device)
        self.gnn = TConvWrapper(self.ndim, self.ndim, num_layers, 4).to(device)
        self.word_encoder = Encoder(self.tokenizer, device, hidden_size=self.enc_dim)
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

    def contextualize_nodes(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor,
                            word_ids: Tensor, word_batch: Tensor, word_starts: Tensor,
                            word_pos: Tensor) -> Tensor:
        node_reprs = self.embed_nodes(atoms)
        edge_attr = self.embed_edges(edge_ids)
        words, ids = to_dense_batch(word_ids, word_batch, fill_value=self.word_encoder.pad_value)
        w_reprs = self.word_encoder(words)[ids][word_starts.eq(1)]
        node_reprs[word_pos] = w_reprs
        node_reprs = self.gnn(node_reprs, edge_index, edge_attr)
        node_reprs[word_pos] += w_reprs
        return node_reprs

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base):
        super(PairClassifier, self).__init__()
        self.base = base
        self.collapser = Sequential(Linear(self.base.ndim * 3, self.base.ndim, False), Dropout(0.5),
                                    LayerNorm(self.base.ndim)).to(base.device)
        self.decoder = LSTM(input_size=self.base.ndim, hidden_size=self.base.ndim, batch_first=True,
                            bidirectional=True).to(base.device)
        self.classifier = Sequential(Linear(self.base.ndim * 8, self.base.ndim * 4), Dropout(0.5), GELU(),
                                     Linear(self.base.ndim * 4, self.base.ndim), Dropout(0.5), GELU(),
                                     Linear(self.base.ndim, 3)).to(base.device)
        self.dropout = Dropout(0.5)

    def readout(self, atoms: Tensor, edge_index: Tensor, edge_ids: Tensor, word_pos: Tensor, word_batch: Tensor,
                word_ids: Tensor, word_starts: Tensor, batch: Tensor) -> Tensor:
        ctx = self.base.contextualize_nodes(atoms, edge_index, edge_ids, word_ids, word_batch, word_starts, word_pos)
        ctx, ids = to_dense_batch(ctx[word_pos], word_batch[word_starts.eq(1)], fill_value=0)
        return ctx

    def entail(self, ctx_h: Tensor, ctx_p: Tensor) -> Tensor:
        nwh, nwp = ctx_h.shape[1], ctx_p.shape[1]
        mask_h = ctx_h.eq(0).all(dim=-1)                                                                    # B, S1
        mask_p = ctx_p.eq(0).all(dim=-1)                                                                    # B, S2
        mask = mask_h.unsqueeze(-1).repeat(1, 1, nwp).bitwise_or(mask_p.unsqueeze(1).repeat(1, nwh, 1))     # B, S1, S2
        cross_atn = bmm(ctx_h, ctx_p.permute(0, 2, 1)).masked_fill_(mask, -1e10)                            # B, S1, S2
        ctx_ph = bmm(cross_atn.softmax(dim=2), ctx_p)                                                       # B, S1, d
        ctx_hp = bmm(cross_atn.softmax(dim=1).permute(0, 2, 1), ctx_h)                                      # B, S2, d
        ctx_h = self.collapser(cat((ctx_h, ctx_h - ctx_ph, ctx_h * ctx_ph), dim=-1))                        # B, S1, d'
        ctx_p = self.collapser(cat((ctx_p, ctx_p - ctx_hp, ctx_p * ctx_hp), dim=-1))                        # B, S2, d'
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
        return self.classifier(cat((ctx_h, ctx_p), dim=-1))

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)
