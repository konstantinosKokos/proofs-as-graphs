from ...typing import List, Tuple, Data, DataLoader, Tensor
from torch_geometric.utils import to_undirected, add_self_loops
import torch


def tensorize_graph(atom_ids: List[int], word_ids: List[int], edge_index: Tuple[List[int], List[int]]) \
        -> Tuple[Tensor, Tensor, Tensor]:
    atom_ids = torch.tensor(atom_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    return (torch.stack((atom_ids, word_ids), dim=-1),
            to_undirected(torch.tensor(edge_index, dtype=torch.long)),
            atom_ids.unsqueeze(-1))



def graph_to_data(atom_ids: List[int], word_ids: List[int], edge_index: Tuple[List[int], List[int]]) -> Data:
    x, edge_index, y = tensorize_graph(atom_ids, word_ids, edge_index)
    return Data(x=x, edge_index=edge_index, y=y)


def graph_loader(data_list: List[Data], batch_size: int) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size)


class GraphPair(Data):
    def __init__(self, edge_index_h: Tensor, x_h: Tensor, edge_index_p: Tensor, x_p: Tensor, y: Tensor):
        super(GraphPair, self).__init__()
        self.edge_index_h = edge_index_h
        self.x_h = x_h
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.y = y

    def __inc__(self, key, value):
        if key == 'edge_index_h':
            return self.x_h.shape(0)
        if key == 'edge_index_p':
            return self.x_p.shape(0)
        return super(GraphPair, self).__inc__(key, value)


def graphs_to_data(graph_a: Tuple[List[int], List[int], Tuple[List[int], List[int]]],
                   graph_b: Tuple[List[int], List[int], Tuple[List[int], List[int]]],
                   label: int) -> GraphPair:
    x_h, edge_index_h, _ = tensorize_graph(*graph_a)
    x_p, edge_index_p, _ = tensorize_graph(*graph_b)
    return GraphPair(edge_index_h, x_h, edge_index_p, x_p, torch.tensor(label, dtype=torch.long))


def pair_loader(data_list: List[GraphPair], batch_size: int) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, follow_batch=['x_h', 'x_p'])