from ..typing import array, Union, Tensor, Module
from torch import rand, tensor, float32
from torch.nn import Parameter, Embedding
from torch.nn.functional import embedding, linear
from torch.nn.init import xavier_normal_


class InvertibleEmbedder(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(InvertibleEmbedder, self).__init__()
        data = rand(num_embeddings, embedding_dim)
        xavier_normal_(data)
        self.table = Parameter(data=data, requires_grad=True)

    def embed(self, ids: Tensor) -> Tensor:
        return embedding(ids, self.table)

    def invert(self, weights: Tensor) -> Tensor:
        return linear(weights, self.table)


def from_table(table: Union[array, Tensor], frozen: bool) -> Embedding:
    ne, ed = table.shape
    embedder = Embedding(ne, ed, padding_idx=0)
    embedder.weight.data = tensor(table).to(float32)
    embedder.requires_grad_(not frozen)
    return embedder
