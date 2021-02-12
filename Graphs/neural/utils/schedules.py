from ...typing import Callable, Optimizer, Logger
from .metrics import mean
from math import cos, radians


def save_if_best(save_fn: Callable[[], None]) -> Callable[[Logger], None]:
    def fn(logger: Logger):
        if mean(logger.stats['dev']['loss'][-1]) == min(map(mean, (logger.stats['dev']['loss']))):
            print(f'Saving..')
            save_fn()
    return fn


def make_cyclic_triangular_schedule(max_lr: float, warmup_steps: int, triangle_decay: int, decay_over: int) \
        -> Callable[[int], float]:
    linear_factor = max_lr / warmup_steps
    cos_window = make_cosine_window(max_lr, warmup_steps, decay_over - warmup_steps)

    def schedule(step: int):
        if step < warmup_steps:
            return linear_factor * step
        num_triangles = (step - warmup_steps) // triangle_decay
        init_step = num_triangles * triangle_decay
        init_lr = cos_window(init_step + warmup_steps)
        down_factor = init_lr / triangle_decay
        return init_lr - down_factor * ((step - warmup_steps) % triangle_decay)
    return schedule


def make_cosine_window(max_lr: float, offset: int, decay_over: int) -> Callable[[int], float]:
    f = 90 / decay_over
    b = - f * offset

    def schedule(step: int) -> float:
        angle = f * step + b
        return cos(radians(angle)) * max_lr

    return schedule


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
