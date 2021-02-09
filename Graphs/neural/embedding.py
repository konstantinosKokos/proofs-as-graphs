from ..typing import array, Union, Tensor, Module
from torch import rand, tensor, float32
from torch.nn import Parameter, Embedding
from torch.nn.functional import embedding, linear


class InvertibleEmbedder(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super(InvertibleEmbedder, self).__init__()
        data = rand(num_embeddings, embedding_dim)
        data[0] = 0.
        self.table = Parameter(data=data, requires_grad=True)
        self.padding_idx = padding_idx

    def embed(self, ids: Tensor) -> Tensor:
        return embedding(ids, self.table, self.padding_idx)

    def invert(self, weights: Tensor) -> Tensor:
        return linear(weights, self.table)


def from_table(table: Union[array, Tensor], frozen: bool) -> Embedding:
    ne, ed = table.shape
    embedder = Embedding(ne, ed, padding_idx=0)
    embedder.weight.data = tensor(table).to(float32)
    embedder.requires_grad_(not frozen)
    return embedder
