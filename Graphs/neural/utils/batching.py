from ...types import Union, List, Tuple, Tensor, GraphData, ProofPair
import torch
from torch_geometric.data import DataLoader


def longt(x: Union[List[int], Tuple[List[int], ...]]) -> Tensor:
    return torch.tensor(x, dtype=torch.long)


def tensorize_graph(graph: GraphData) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    node_labels = longt(graph.nodes)
    edge_idx = longt(graph.edge_index)
    edge_labels = longt(graph.edge_attrs)
    anchors = longt(graph.roots)
    word_ids, word_starts = list(zip(*graph.words))
    return node_labels, edge_idx, edge_labels, anchors, longt(word_ids), longt(word_starts)


def graphs_to_data(premise_graph: GraphData, hypothesis_graph: GraphData, label: int) -> ProofPair:
    return ProofPair(tensorize_graph(premise_graph), tensorize_graph(hypothesis_graph), longt([label]))


def pair_loader(pairs: List[ProofPair], batch_size: int, shuffle: bool):
    return DataLoader(pairs, batch_size=batch_size, follow_batch=['x_h', 'x_p', 'word_ids_h', 'word_ids_p'],
                      shuffle=shuffle)

