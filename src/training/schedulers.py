"""Two-level learning rate scheduler: warmup LR then constraint LR."""


class LearningRateScheduler:
    """Drops LR from warmup_lr to drop_lr at warmup_epochs boundary."""

    def __init__(self, optimizer, warmup_lr=1e-3, drop_lr=1e-5,
                 warmup_epochs=50, **kwargs):
        self.optimizer = optimizer
        self.warmup_lr = warmup_lr
        self.drop_lr = drop_lr
        self.warmup_epochs = warmup_epochs
        self._current_lr = warmup_lr

    def step(self, epoch):
        lr = self.warmup_lr if epoch < self.warmup_epochs else self.drop_lr
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        self._current_lr = lr
        return lr

    def get_lr(self):
        return self._current_lr
