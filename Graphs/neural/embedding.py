from ..typing import array, Union, Tensor, Module
from torch import rand, tensor, float32
from torch.nn import Parameter, Embedding
from torch.nn.functional import embedding, linear


class InvertibleEmbedder(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int = 0):
        super(InvertibleEmbedder, self).__init__()
        self.table = Parameter(data=rand(num_embeddings, embedding_dim), requires_grad=True)
        self.padding_idx = padding_idx

    def embed(self, ids: Tensor) -> Tensor:
        return embedding(ids, self.table, self.padding_idx)

    def invert(self, weights: Tensor) -> Tensor:
        return linear(weights, self.table.t())


def from_table(table: Union[array, Tensor]) -> Embedding:
    ne, ed = table.shape
    embedder = Embedding(ne, ed, padding_idx=0)
    embedder.weight.data = tensor(table).to(float32)
    return embedder
