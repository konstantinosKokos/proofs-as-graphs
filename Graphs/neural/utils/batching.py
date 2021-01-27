from ...typing import List, Tuple, Data, DataLoader
import torch


def graph_to_data(atom_ids: List[int], word_ids: List[int], edge_index: Tuple[List[int], List[int]]) -> Data:
    atom_ids = torch.tensor(atom_ids, dtype=torch.long)
    word_ids = torch.tensor(word_ids, dtype=torch.long)
    return Data(x=torch.stack((atom_ids, word_ids), dim=-1), edge_index=torch.tensor(edge_index, dtype=torch.long),
                y=atom_ids.unsqueeze(-1))


def make_dataloader(data_list: List[Data], batch_size: int) -> DataLoader:
    return DataLoader(data_list, batch_size=batch_size)
