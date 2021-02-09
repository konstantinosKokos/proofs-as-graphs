from ...typing import Callable, Optimizer, Logger
from .metrics import mean


def save_if_best(save_fn: Callable[[], None]) -> Callable[[Logger], None]:
    def fn(logger: Logger):
        if mean(logger.stats['dev']['loss'][-1]) == min(map(mean, (logger.stats['dev']['loss']))):
            print(f'Saving..')
            save_fn()
    return fn


def exponential_decay(init_lr: float, decay: float, warmup: int = 0) -> Callable[[int], float]:
    def schedule(step: int) -> float:
        if step < warmup:
            return step / warmup * init_lr
        return init_lr * decay ** (step - warmup)
    return schedule


class Scheduler:
    def __init__(self, opt: Optimizer, schedule: Callable[[int], float]):
        self.opt = opt
        self.schedule = schedule
        self.step_num = 0
        self.lr = 0.

    def step(self) -> None:
        self.step_num += 1
        self.lr = self.schedule(self.step_num)
        for i, p in enumerate(self.opt.param_groups):
            p['lr'] = self.lr
        self.opt.step()

    def zero_grad(self) -> None:
        self.opt.zero_grad()
