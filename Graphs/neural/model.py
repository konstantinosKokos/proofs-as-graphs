from torch_geometric.nn import global_max_pool, MessagePassing
from torch_geometric.utils import to_dense_batch
from ..typing import Tensor, Dict, array
from ..neural.embedding import InvertibleEmbedder, from_table
from ..neural.mha import MultiHeadAttention
from torch.nn import Module, ModuleList, Linear, Sequential, Dropout, GELU, GRU
from torch import cat, save


class GNN(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super(GNN, self).__init__()
        self.mlp = Sequential(Linear(in_channels * 2, out_channels), GELU())

    def forward(self, x: Tensor, edge_index: Tensor):
        # row, col = edge_index
        # deg = degree(col, x.shape[0], dtype=x.dtype)
        return self.propagate(edge_index, x=x)  #, norm=deg.pow(-1)[col])

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.mlp(cat((x_i, x_j), dim=-1))  # * norm.unsqueeze(-1)


class Base(Module):
    def __init__(self, atom_map: Dict[str, int], embedding_table: array, num_layers: int, device: str = 'cuda'):
        super(Base, self).__init__()
        self.gru = GRU(input_size=300, hidden_size=150, bidirectional=True, batch_first=True).to(device)
        self.word_embedding = Sequential(from_table(embedding_table, False)).to(device)
        self.atom_embedding = InvertibleEmbedder(len(atom_map), 300).to(device)
        self.r2c = ModuleList([GNN(300, 150) for _ in range(num_layers)]).to(device)
        self.c2r = ModuleList([GNN(300, 150) for _ in range(num_layers)]).to(device)
        self.layerwise_aggr = Sequential(Linear(300, 300), GELU()).to(device)
        self.atom_map = atom_map
        self.device = device
        self.dropout = Dropout(0.3)

    def embed_atoms(self, atoms: Tensor) -> Tensor:
        return self.atom_embedding.embed(atoms).squeeze(1)

    def embed_words(self, words: Tensor, w_batch: Tensor) -> Tensor:
        sents, ids = to_dense_batch(self.word_embedding(words), w_batch, fill_value=0)
        return self.gru(sents)[0][ids.eq(True)]

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def embed_nodes(self, atoms: Tensor, words: Tensor, w_pos: Tensor, w_batch: Tensor) -> Tensor:
        atom_vectors = self.embed_atoms(atoms)
        word_vectors = self.embed_words(words, w_batch)
        atom_vectors.index_put_((w_pos,), word_vectors)
        return atom_vectors

    def contextualize_nodes(self, atoms: Tensor, words: Tensor, w_pos: Tensor, w_batch: Tensor,
                            edge_index: Tensor) -> Tensor:
        vectors = self.embed_nodes(atoms, words, w_pos, w_batch)
        for r2c, c2r in zip(self.r2c, self.c2r):
            vectors = (vectors +
                       self.layerwise_aggr(cat([r2c(vectors, edge_index), c2r(vectors, edge_index.flip(0))], dim=-1)))
            vectors = self.dropout(vectors)
        return vectors

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)


class PairClassifier(Module):
    def __init__(self, base: Base, device: str):
        super(PairClassifier, self).__init__()
        self.base = base
        self.mha_source = MultiHeadAttention(1, 300, 300, 300, 300, 300, 300, 0.1).to(device)
        self.mha_target = MultiHeadAttention(1, 300, 300, 300, 300, 300, 300, 0.1).to(device)
        self.combiner = Sequential(Linear(300*4, 300*2), GELU(), Dropout(0.15), Linear(300*2, 3)).to(device)
        # self.expander = Sequential(Linear(512, 512),  GELU(), Dropout(0.15), Linear(512, 1024),).to(device)
        # self.combiner = Sequential(Linear(2048, 512), GELU(), Dropout(0.15),
        #                            Linear(512, 128), GELU(), Dropout(0.15), Linear(128, 3)).to(device)
        self.device = device

    def readout(self, atoms: Tensor, words: Tensor, w_pos: Tensor, w_batch: Tensor,
                edge_index: Tensor, batch: Tensor) -> Tensor:
        vectors = self.base.contextualize_nodes(atoms, words, w_pos, w_batch, edge_index)
        return to_dense_batch(vectors, batch)[0]
        # # vectors = self.expander(vectors)
        # return global_max_pool(vectors, batch)

    def entail(self, x1: Tensor, x2: Tensor):
        s = self.mha_source(x2, x1, x1).max(dim=1)[0]
        t = self.mha_target(x1, x2, x2).max(dim=1)[0]
        x = cat((s, t, s * t, s - t), dim=-1)
        return self.combiner(x)

    def save(self, path: str) -> None:
        save({'model_state_dict': self.state_dict()}, path)