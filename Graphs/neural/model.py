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

    def embed_nodes(self, x: Tensor) -> Tensor:
        atom_ids, word_ids = x.chunk(2, dim=-1)
        atom_vectors = self.atom_embedding.embed(atom_ids)
        word_vectors = self.word_embedding(word_ids)
        return (atom_vectors + word_vectors).squeeze()

    def contextualize_nodes(self, x: Tensor, edge_index: Tensor, num_hops: int) -> Tensor:
        vectors = self.embed_nodes(x)
        for _ in range(num_hops):
            vectors = self.cgn(x=x, edge_index=edge_index)
        return vectors

    def embed_graphs(self, x: Tensor, edge_index: Tensor, batch: Tensor, num_hops: int, pool: str) -> Tensor:
        vectors = self.contextualize_nodes(x, edge_index, num_hops)
        if pool == 'mean':
            return global_mean_pool(vectors, batch)
        if pool == 'max':
            return global_max_pool(vectors, batch)
        raise ValueError
