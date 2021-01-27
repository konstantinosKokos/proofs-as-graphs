import torch
from torch.nn.functional import embedding, linear
from torch.nn import Module, Parameter, Embedding
from typing import Union
from numpy import array


class InvertibleEmbedder(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super(InvertibleEmbedder, self).__init__()
        self.table = Parameter(data=torch.rand(num_embeddings, embedding_dim), requires_grad=True)
        self.padding_idx = padding_idx

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return embedding(ids, self.table, self.padding_idx)

    def invert(self, weights: torch.Tensor) -> torch.Tensor:
        return linear(weights, self.table.t())


def from_table(table: Union[array, torch.Tensor]) -> Embedding:
    ne, ed = table.shape
    embedder = Embedding(ne, ed, padding_idx=0)
    embedder.weight.data = torch.tensor(table).to(torch.float)
    return embedder
