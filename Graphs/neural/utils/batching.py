from ...typing import List, Tuple, Data, DataLoader, Tensor
from torch_geometric.utils import to_undirected, add_self_loops
import torch


def tensorize_graph(atom_ids: List[int], word_ids: List[int], word_pos: List[int],
                    edge_index: Tuple[List[int], List[int]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    atoms = torch.tensor(atom_ids, dtype=torch.long)
    return (torch.tensor(atom_ids, dtype=torch.long),
            torch.tensor(word_ids, dtype=torch.long),
            torch.tensor(word_pos, dtype=torch.long),
            torch.tensor(edge_index, dtype=torch.long),
            atoms.unsqueeze(-1))


def graph_to_data(atom_ids: List[int], word_ids: List[int], word_pos: List[int],
                  edge_index: Tuple[List[int], List[int]]) -> Data:
    atom_ids, word_ids, word_pos, edge_index, y = tensorize_graph(atom_ids, word_ids, word_pos, edge_index)
    return ProofData(x=atom_ids, word_ids=word_ids, word_pos=word_pos, edge_index=edge_index, y=y)


def graph_loader(data_list: List[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, follow_batch=['word_ids'], shuffle=shuffle)


class ProofData(Data):
    def __init__(self, edge_index: Tensor, x: Tensor, word_ids: Tensor, word_pos: Tensor, y: Tensor):
        super(ProofData, self).__init__()
        self.edge_index = edge_index
        self.x = x
        self.word_ids = word_ids
        self.word_pos = word_pos
        self.y = y

    def __inc__(self, key, value):
        if key == 'word_pos':
            return self.num_nodes
        return super(ProofData, self).__inc__(key, value)


class ProofPair(Data):
    def __init__(self, edge_index_h: Tensor, x_h: Tensor, word_ids_h: Tensor, word_pos_h: Tensor,
                 edge_index_p: Tensor, x_p: Tensor, word_ids_p: Tensor, word_pos_p: Tensor, y: Tensor):
        super(ProofPair, self).__init__()
        self.edge_index_h = edge_index_h
        self.edge_index_p = edge_index_p
        self.x_h = x_h
        self.x_p = x_p
        self.word_ids_h = word_ids_h
        self.word_ids_p = word_ids_p
        self.word_pos_h = word_pos_h
        self.word_pos_p = word_pos_p
        self.y = y

    def __inc__(self, key, value):
        if key in {'edge_index_h', 'word_pos_h'}:
            return self.x_h.shape[0]
        if key in {'edge_index_p', 'word_pos_p'}:
            return self.x_p.shape[0]
        return super(ProofPair, self).__inc__(key, value)


def graphs_to_data(graph_h: Tuple[List[int], List[int], List[int], Tuple[List[int], List[int]]],
                   graph_p: Tuple[List[int], List[int], List[int], Tuple[List[int], List[int]]],
                   label: int) -> ProofPair:
    atoms_h, words_h, word_pos_h, edge_index_h, _ = tensorize_graph(*graph_h)
    atoms_p, words_p, word_pos_p, edge_index_p, _ = tensorize_graph(*graph_p)
    return ProofPair(edge_index_h=edge_index_h, edge_index_p=edge_index_p, word_ids_h=words_h, word_ids_p=words_p,
                     x_h=atoms_h, x_p=atoms_p, word_pos_h=word_pos_h, word_pos_p=word_pos_p,
                     y=torch.tensor(label, dtype=torch.long))


def pair_loader(data_list: List[ProofPair], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, follow_batch=['x_h', 'x_p', 'word_ids_h', 'word_ids_p'],
                      shuffle=shuffle)