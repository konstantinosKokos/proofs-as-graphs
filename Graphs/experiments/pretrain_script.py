from .pretrain import *
from ..data.process_lassy import load_lassy
from ..neural.utils.batching import graph_loader, graph_to_data
from ..neural.utils.schedules import exponential_decay, Scheduler
from ..typing import array
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def init_pretrain() -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Base, PretrainLogger, array]:
    (train, dev, test), atom_map, table = load_lassy()

    train_dl = graph_loader([graph_to_data(graph) for graph in train], 32, True)
    dev_dl = graph_loader([graph_to_data(graph) for graph in dev], 1024, False)
    test_dl = graph_loader([graph_to_data(graph) for graph in test], 1024, False)
    model = Base(atom_map, 6)
    opt = Scheduler(AdamW(model.parameters(), lr=0., weight_decay=1e-02),
                    exponential_decay(init_lr=5e-04, decay=1 - 0.98/len(train_dl), warmup=500))
    logger = PretrainLogger(CrossEntropyLoss(ignore_index=0, reduction='mean'), opt)
    return (train_dl, dev_dl, test_dl), model, logger, table


def load_pretrained(model: Base, path: str = './stored_models/pretrain.model') -> Base:
    tmp = torch.load(path)
    model.load_state_dict(tmp['model_state_dict'])
    return model


def main():
    (train_dl, dev_dl, test_dl), model, logger, _ = init_pretrain()
    mask_chance = 0.25
    for i in range(200):
        print(f' === {i} === ')
        log_epoch(model=model, dataloader=train_dl, mask_chance=mask_chance, logger=logger, train=True)
        print(logger.aggr_epoch(True))
        log_epoch(model=model, dataloader=dev_dl, mask_chance=mask_chance, logger=logger, train=False)
        print(logger.aggr_epoch(False))
