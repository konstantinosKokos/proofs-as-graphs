from ...typing import List, Tuple, Data, DataLoader, Tensor, Union, GraphData
import torch


def longt(x: Union[List[int], Tuple[List[int], ...]]) -> Tensor:
    return torch.tensor(x, dtype=torch.long)


def tensorize_graph(graph: GraphData) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    node_labels = longt(graph.nodes)
    edge_index = longt(graph.edges)
    edge_attr = longt(graph.edge_attrs)
    anchors = longt(graph.roots)
    _word_ids, _word_starts = list(zip(*graph.words))
    conc = longt([graph.conclusion])
    return node_labels, edge_index, edge_attr, anchors, longt(_word_ids), longt(_word_starts), conc


def graph_to_data(graph: GraphData) -> Data:
    node_labels, edge_index, edge_attr, anchors, word_ids, word_starts, _ = tensorize_graph(graph)
    return ProofData(x=node_labels, word_ids=word_ids, word_starts=word_starts, edge_attr=edge_attr,
                     word_pos=anchors, edge_index=edge_index, y=node_labels.unsqueeze(-1))


def graph_loader(data_list: List[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, follow_batch=['word_ids'], shuffle=shuffle)


class ProofData(Data):
    def __init__(self, edge_index: Tensor, x: Tensor, edge_attr: Tensor, word_ids: Tensor,
                 word_starts: Tensor, word_pos: Tensor, y: Tensor):
        super(ProofData, self).__init__()
        self.edge_index = edge_index
        self.x = x
        self.word_ids = word_ids
        self.word_starts = word_starts
        self.word_pos = word_pos
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y

    def __inc__(self, key, value):
        if key == 'word_pos':
            return self.num_nodes
        return super(ProofData, self).__inc__(key, value)


class ProofPair(Data):
    def __init__(self, edge_index_h: Tensor, x_h: Tensor, word_ids_h: Tensor, word_starts_h: Tensor,
                 word_pos_h: Tensor, edge_attr_h: Tensor, conc_h: Tensor,
                 edge_index_p: Tensor, x_p: Tensor, word_ids_p: Tensor, word_starts_p: Tensor, word_pos_p: Tensor,
                 edge_attr_p: Tensor, conc_p: Tensor, y: Tensor):
        super(ProofPair, self).__init__()
        self.edge_index_h = edge_index_h
        self.edge_index_p = edge_index_p
        self.x_h = x_h
        self.x_p = x_p
        self.word_ids_h = word_ids_h
        self.word_ids_p = word_ids_p
        self.word_starts_h = word_starts_h
        self.word_starts_p = word_starts_p
        self.word_pos_h = word_pos_h
        self.word_pos_p = word_pos_p
        self.y = y
        self.conc_h = conc_h
        self.conc_p = conc_p
        self.edge_attr_h = edge_attr_h
        self.edge_attr_p = edge_attr_p

    def __inc__(self, key, value):
        if key in {'edge_index_h', 'word_pos_h', 'conc_h'}:
            return self.x_h.shape[0]
        if key in {'edge_index_p', 'word_pos_p', 'conc_p'}:
            return self.x_p.shape[0]
        return super(ProofPair, self).__inc__(key, value)


def graphs_to_data(graph_h: GraphData, graph_p: GraphData, label: int) -> ProofPair:

    node_labels_h, edge_index_h, edge_attr_h, anchors_h, word_ids_h, word_starts_h, conc_h = tensorize_graph(graph_h)
    node_labels_p, edge_index_p, edge_attr_p, anchors_p, word_ids_p, word_starts_p, conc_p = tensorize_graph(graph_p)
    return ProofPair(edge_index_h=edge_index_h, edge_index_p=edge_index_p,
                     word_ids_h=word_ids_h, word_ids_p=word_ids_p,
                     x_h=node_labels_h, x_p=node_labels_p, word_pos_h=anchors_h, word_pos_p=anchors_p,
                     word_starts_h=word_starts_h, word_starts_p=word_starts_p,
                     edge_attr_h=edge_attr_h, edge_attr_p=edge_attr_p, conc_h=conc_h, conc_p=conc_p,
                     y=longt([label]))


def pair_loader(data_list: List[ProofPair], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size, follow_batch=['x_h', 'x_p', 'word_ids_h', 'word_ids_p'],
                      shuffle=shuffle)