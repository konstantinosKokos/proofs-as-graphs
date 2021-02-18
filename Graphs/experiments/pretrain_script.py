from .pretrain import *
from ..data.process_lassy import load_lassy
from ..data.tokenizer import load_tokenizer
from ..neural.utils.batching import graph_loader, graph_to_data
from ..neural.utils.schedules import exponential_decay, Scheduler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW


def init_pretrain(which: str) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Base, PretrainLogger]:
    (train, dev, test) = load_lassy(which)
    tokenizer = load_tokenizer(which)
    train_dl = graph_loader([graph_to_data(graph) for graph in train], 16, True)
    dev_dl = graph_loader([graph_to_data(graph) for graph in dev], 32, False)
    test_dl = graph_loader([graph_to_data(graph) for graph in test], 32, False)
    model = Base(tokenizer, 10, 'cuda')
    opt = Scheduler(AdamW(model.parameters(), lr=0., weight_decay=1e-02),
                    exponential_decay(init_lr=5e-04, decay=1 - 0.98/len(train_dl), warmup=500))
    logger = PretrainLogger(CrossEntropyLoss(ignore_index=0, reduction='mean'), opt)
    return (train_dl, dev_dl, test_dl), model, logger


def load_pretrained(model: Base, path: str = './stored_models/pretrain.model') -> Base:
    tmp = torch.load(path)
    model.load_state_dict(tmp['model_state_dict'])
    return model


def main(encoder: str = 'spacy'):
    (train_dl, dev_dl, test_dl), model, logger = init_pretrain(encoder)
    mask_chance = 0.85
    for i in range(200):
        print(f' === {i} === ')
        log_epoch(model=model, dataloader=train_dl, mask_chance=mask_chance, logger=logger, train=True)
        print(logger.aggr_epoch(True))
        log_epoch(model=model, dataloader=dev_dl, mask_chance=mask_chance, logger=logger, train=False)
        print(logger.aggr_epoch(False))
