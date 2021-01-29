import torch
from ..model import Model
from ...typing import Batch, Tensor, Tuple, Logger, DataLoader, Module, OptLike
from ..utils.metrics import accuracy, mean


def mask_and_predict(model: Model, batch: Batch, mask_chance: float, num_hops: int) -> Tuple[Tensor, Tensor]:
    atoms, words = batch.x.chunk(2, dim=-1)
    amask = make_mask(atoms, mask_chance)
    truths = atoms[amask.eq(1)].squeeze(-1)
    atoms[amask.eq(1)] = 0
    wmask = make_mask(words, mask_chance)
    words[wmask.eq(1)] = 0
    amask = amask.squeeze(-1)
    atom_reprs = model.contextualize_nodes(torch.cat((atoms, words), dim=-1), batch.edge_index, num_hops)
    predictions = model.classify_nodes(atom_reprs[amask.eq(1)])
    return predictions, truths


@torch.no_grad()
def make_mask(atoms: Tensor, mask_chance: float) -> Tensor:
    non_padded = atoms.ne(0)
    r = torch.rand_like(non_padded, dtype=torch.float).lt(mask_chance)
    return r.bitwise_and(non_padded)


def log_batch(model: Model, batch: Batch, mask_chance: float, num_hops: int, logger: Logger, train: bool) -> None:
    predictions, truths = mask_and_predict(model, batch, mask_chance, num_hops)
    logger.log(predictions, truths, train=train)


def log_epoch(model: Model, dataloader: DataLoader, mask_chance: float, num_hops: int, logger: Logger, train: bool):
    for batch in iter(dataloader):
        log_batch(model, batch, mask_chance, num_hops, logger, train)
    logger.register_epoch(train)


class PretrainLogger(Logger):
    def __init__(self, loss_fn: Module, opt: OptLike):
        self.loss_fn = loss_fn
        self.opt = opt
        self.stats = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                      'dev': {'loss': [], 'nsamples': [], 'correct': []}}
        self.cur_epoch = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                          'dev': {'loss': [], 'nsamples': [], 'correct': []}}

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

    def register_epoch(self, train: bool):
        if train:
            for k, v in self.cur_epoch['train'].items():
                self.stats['train'][k].append(v)
            self.cur_epoch['train'] = {'loss': [], 'nsamples': [], 'correct': []}
        else:
            for k, v in self.cur_epoch['dev'].items():
                self.stats['dev'][k].append(v)
            self.cur_epoch['dev'] = {'loss': [], 'nsamples': [], 'correct': []}

    def aggr_last_epoch(self, train: bool) -> Tuple[float, float]:
        k = 'train' if train else 'dev'
        return mean(self.stats[k]['loss'][-1]), sum(self.stats[k]['correct'][-1]) / sum(self.stats[k]['nsamples'][-1])
