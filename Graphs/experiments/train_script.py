import torch
from .train import *
from .pretrain_script import init_untrained, load_pretrained
from ..data.process_sick import load_sick
from ..neural.utils.batching import pair_loader, graphs_to_data
from ..neural.utils.schedules import make_cyclic_triangular_schedule, Scheduler
from torch.nn import CrossEntropyLoss
from adabelief_pytorch import AdaBelief

torch.manual_seed(42)


def init_train(which: str) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], PairClassifier, TrainLogger]:
    base = init_untrained(which, num_layers=8, weight_sharing=False, device='cuda')
    # base = load_pretrained(base)
    model = PairClassifier(base)
    train, dev, test = load_sick(which)
    train_dl = pair_loader([graphs_to_data(*sample) for sample in train], 16, True)
    dev_dl = pair_loader([graphs_to_data(*sample) for sample in dev], 64, False)
    test_dl = pair_loader([graphs_to_data(*sample) for sample in test], 64, False)

    logger = TrainLogger(CrossEntropyLoss(ignore_index=-100, reduction='none',
                                          # weight=torch.tensor([1.2, 0.6, 2.1], device='cuda')
                                          ),
                         Scheduler(
                             AdaBelief(model.parameters(), lr=2e-04, weight_decay=1e-02, eps=1e-12,
                                       weight_decouple=True, print_change_log=False),
                             make_cyclic_triangular_schedule(max_lr=2e-04, warmup_steps=250,
                                                             triangle_decay=len(train_dl) * 1,
                                                             decay_over=len(train_dl) * 90))
                         )

    return (train_dl, dev_dl, test_dl), model, logger


def main(which: str):
    (train_dl, dev_dl, test_dl), model, logger = init_train(which)
    for i in range(500):
        print(f' === {i} === ')
        log_epoch(model=model, dataloader=train_dl, logger=logger, train=True)
        print(logger.aggr_last_epoch(True))
        log_epoch(model=model, dataloader=dev_dl, logger=logger, train=False)
        print(logger.aggr_last_epoch(False))
        print('\n')
