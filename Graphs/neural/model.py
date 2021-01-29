from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.nn import Module
from torch import Tensor
from ..neural.embedding import InvertibleEmbedder, from_table


class Model(Module):
    def __init__(self, atom_map, embedding_table):
        super(Model, self).__init__()
        self.word_embedding = from_table(embedding_table)
        self.atom_embedding = InvertibleEmbedder(len(atom_map) + 1, 300)
        self.cgn = GCNConv(in_channels=300, out_channels=300)

    def embed_atoms(self, atoms: Tensor) -> Tensor:
        return self.atom_embedding.embed(atoms).squeeze(1)

    def embed_words(self, words: Tensor) -> Tensor:
        return self.word_embedding(words).squeeze(1)

    def classify_nodes(self, weights: Tensor) -> Tensor:
        return self.atom_embedding.invert(weights)

    def embed_nodes(self, x: Tensor) -> Tensor:
        atoms, words = x.chunk(2, dim=-1)
        return self.embed_atoms(atoms) + self.embed_words(words)

    def contextualize_nodes(self, x: Tensor, edge_index: Tensor, num_hops: int) -> Tensor:
        vectors = self.embed_nodes(x)
        for _ in range(num_hops):
            vectors = self.cgn(x=vectors, edge_index=edge_index)
        return vectors

    def embed_graphs(self, x: Tensor, edge_index: Tensor, batch: Tensor, num_hops: int, pool: str) -> Tensor:
        vectors = self.contextualize_nodes(x, edge_index, num_hops)
        if pool == 'mean':
            return global_mean_pool(vectors, batch)
        if pool == 'max':
            return global_max_pool(vectors, batch)
        raise ValueError
