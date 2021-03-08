import torch
from ..neural.model import PairClassifier
from ..typing import Batch, Tensor, Tuple, Logger, DataLoader, Module, OptLike, Callable
from ..neural.utils.metrics import mean, per_class_accuracy
from ..neural.utils.schedules import save_if_best


from tqdm import tqdm


def get_entailment_values(model: PairClassifier, batch: Batch) -> Tuple[Tensor, Tensor, Tensor]:
    vectors_h = model.readout(atoms=batch.x_h, word_ids=batch.word_ids_h, word_pos=batch.word_pos_h,
                              word_batch=batch.word_ids_h_batch, word_starts=batch.word_starts_h,
                              edge_index=batch.edge_index_h, edge_ids=batch.edge_attr_h, batch=batch.x_h_batch)
    vectors_p = model.readout(atoms=batch.x_p, word_ids=batch.word_ids_p, word_pos=batch.word_pos_p,
                              word_batch=batch.word_ids_p_batch, word_starts=batch.word_starts_p,
                              edge_index=batch.edge_index_p, edge_ids=batch.edge_attr_p, batch=batch.x_p_batch)
    predictions = model.entail(vectors_h, vectors_p)
    return predictions, batch.y, batch.w


def log_batch(model: PairClassifier, batch: Batch, logger: Logger, train: bool) -> None:
    predictions, truths, w = get_entailment_values(model, batch.to(model.base.device))
    logger.log(predictions, truths, w, train=train)


def log_epoch(model: PairClassifier, dataloader: DataLoader, logger: Logger, train: bool):
    model.train(train)
    for batch in tqdm(dataloader):
        log_batch(model, batch, logger, train)
    logger.register_epoch(train, save_if_best(lambda: model.save(f'./Graphs/io/{model.base.tokenizer.name}/train.model')))


class TrainLogger(Logger):
    def __init__(self, loss_fn: Module, opt: OptLike):
        self.loss_fn = loss_fn
        self.opt = opt
        self.stats = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                      'dev': {'loss': [], 'nsamples': [], 'correct': []}}
        self.cur_epoch = {'train': {'loss': [], 'nsamples': [], 'correct': []},
                          'dev': {'loss': [], 'nsamples': [], 'correct': []}}

    def log(self, *tensors: Tensor, train: bool):
        predictions, truths, w = tensors
        if train:
            self.train_log(predictions, truths, w)
        else:
            self.dev_log(predictions, truths, w)

    def train_log(self, predictions: Tensor, truths: Tensor, w: Tensor):
        loss = (self.loss_fn(predictions, truths) * w).sum()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        total, correct = per_class_accuracy(predictions, truths)
        self.cur_epoch['train']['loss'].append(loss.item())
        self.cur_epoch['train']['nsamples'].append(total)
        self.cur_epoch['train']['correct'].append(correct)

    @torch.no_grad()
    def dev_log(self, predictions: Tensor, truths: Tensor):
        loss = self.loss_fn(predictions, truths)
        total, correct = per_class_accuracy(predictions, truths)
        self.cur_epoch['dev']['loss'].append(loss.item())
        self.cur_epoch['dev']['nsamples'].append(total)
        self.cur_epoch['dev']['correct'].append(correct)

    def register_epoch(self, train: bool, callback: Callable[['TrainLogger'], None]):
        if train:
            for k, v in self.cur_epoch['train'].items():
                self.stats['train'][k].append(v)
            self.cur_epoch['train'] = {'loss': [], 'nsamples': [], 'correct': []}
        else:
            for k, v in self.cur_epoch['dev'].items():
                self.stats['dev'][k].append(v)
            self.cur_epoch['dev'] = {'loss': [], 'nsamples': [], 'correct': []}
            callback(self)

    def aggr_last_epoch(self, train: bool) -> Tuple[float, ...]:
        k = 'train' if train else 'dev'
        loss = mean(self.stats[k]['loss'][-1])
        pcc = list(map(sum, zip(*self.stats[k]['correct'][-1])))
        pct = list(map(sum, zip(*self.stats[k]['nsamples'][-1])))
        pca = tuple(map(lambda x, y: x/y, pcc + [sum(pcc)], pct + [sum(pct)]))
        return (loss,) + pca
