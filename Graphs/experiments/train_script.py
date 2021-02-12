import torch
from .train import *
from .pretrain_script import init_pretrain, load_pretrained
from ..data.tokenizer import load_tokenizer
from ..data.process_sick import load_sick
from ..neural.utils.batching import pair_loader, graphs_to_data
from ..neural.utils.schedules import make_cyclic_triangular_schedule, Scheduler
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW, Adam


def init_train(which: str) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], PairClassifier, TrainLogger]:
    _, base, _ = init_pretrain(which)
    # base = load_pretrained(base)
    tokenizer = load_tokenizer(which)
    model = PairClassifier(base, tokenizer)
    train, dev, test = load_sick(which)
    train_dl = pair_loader([graphs_to_data(*sample) for sample in train], 32, True)
    dev_dl = pair_loader([graphs_to_data(*sample) for sample in dev], 64, False)
    test_dl = pair_loader([graphs_to_data(*sample) for sample in test], 64, False)
    logger = TrainLogger(CrossEntropyLoss(ignore_index=-100, reduction='sum'),
                         Scheduler(Adam(model.parameters(), lr=0),  #, weight_decay=0.01),
                                   make_cyclic_triangular_schedule(max_lr=1e-4, warmup_steps=len(train_dl),
                                                                   triangle_decay=len(train_dl),
                                                                   decay_over=len(train_dl) * 90)))
    return (train_dl, dev_dl, test_dl), model, logger


def main(which: str):
    (train_dl, dev_dl, test_dl), model, logger = init_train(which)
    for i in range(500):
        print(f' === {i} === ')
        log_epoch(model=model, dataloader=train_dl, logger=logger, train=True, pause=i > 5)
        print(logger.aggr_last_epoch(True))
        log_epoch(model=model, dataloader=test_dl, logger=logger, train=False, pause=False)
        print(logger.aggr_last_epoch(False))
        print('\n')
