import torch
from torch import where
from ..neural.model import Base
from ..typing import Batch, Tensor, Tuple, Logger, DataLoader, Module, OptLike, Maybe, Callable
from ..neural.utils.metrics import accuracy, mean
from ..neural.utils.schedules import save_if_best


def mask_and_predict(model: Base, batch: Batch, mask_chance: float, ) -> Tuple[Tensor, Tensor]:
    amask = make_mask(batch.x, mask_chance, model.atom_map['[PAD]'])
    truths = batch.x[amask.eq(1)].squeeze(-1)
    atoms = where(amask.eq(1), model.atom_map['[MASK]'], batch.x)

    amask = amask.squeeze(-1)
    vectors = model.contextualize_nodes(atoms=atoms, edge_index=batch.edge_index, edge_ids=batch.edge_attr)
    predictions = model.classify_nodes(vectors[amask.eq(1)])
    return predictions, truths


@torch.no_grad()
def make_mask(atoms: Tensor, mask_chance: float, ignore: int) -> Tensor:
    non_padded = atoms.ne(ignore)
    r = torch.rand_like(non_padded, dtype=torch.float).lt(mask_chance)
    return r.bitwise_and(non_padded)


def log_batch(model: Base, batch: Batch, mask_chance: float, logger: Logger, train: bool) -> None:
    predictions, truths = mask_and_predict(model, batch.to(model.device), mask_chance)
    logger.log(predictions, truths, train=train)


def log_epoch(model: Base, dataloader: DataLoader, mask_chance: float, logger: Logger, train: bool):
    model.train(train)
    for batch in iter(dataloader):
        log_batch(model, batch, mask_chance, logger, train)
    logger.register_epoch(train, save_if_best(lambda: model.save('./stored_models/pretrain.model')))


class PretrainLogger(Logger):
    def __init__(self, loss_fn: Module, opt: OptLike, save_to: Maybe[str] = None):
        self.loss_fn = loss_fn
        self.opt = opt
        self.stats = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                      'dev': {'loss': [], 'nsamples': [], 'correct': []}}
        self.cur_epoch = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                          'dev': {'loss': [], 'nsamples': [], 'correct': []}}
        self.save_to = save_to

    def log(self, *tensors: Tensor, train: bool):
        predictions, truths = tensors
        if train:
            self.train_log(predictions, truths)
        else:
            self.dev_log(predictions, truths)

    def train_log(self, predictions: Tensor, truths: Tensor):
        loss = self.loss_fn(predictions, truths)
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        total, correct = accuracy(predictions, truths)
        self.cur_epoch['train']['loss'].append(loss.item())
        self.cur_epoch['train']['nsamples'].append(total)
        self.cur_epoch['train']['correct'].append(correct)

    @torch.no_grad()
    def dev_log(self, predictions: Tensor, truths: Tensor):
        loss = self.loss_fn(predictions, truths)
        total, correct = accuracy(predictions, truths)
        self.cur_epoch['dev']['loss'].append(loss.item())
        self.cur_epoch['dev']['nsamples'].append(total)
        self.cur_epoch['dev']['correct'].append(correct)

    def register_epoch(self, train: bool, callback: Callable[['PretrainLogger'], None]):
        if train:
            for k, v in self.cur_epoch['train'].items():
                self.stats['train'][k].append(v)
            self.cur_epoch['train'] = {'loss': [], 'nsamples': [], 'correct': []}
        else:
            for k, v in self.cur_epoch['dev'].items():
                self.stats['dev'][k].append(v)
            self.cur_epoch['dev'] = {'loss': [], 'nsamples': [], 'correct': []}
            callback(self)

    def aggr_epoch(self, train: bool, epoch: Maybe[int] = None) -> Tuple[float, float]:
        e = -1 if epoch is None else epoch
        k = 'train' if train else 'dev'
        return mean(self.stats[k]['loss'][e]), sum(self.stats[k]['correct'][e]) / sum(self.stats[k]['nsamples'][e])
