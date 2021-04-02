from torch.optim.lr_scheduler import MultiStepLR

class LinearWarmupScheduler(MultiStepLR):
    """ Linearly warm-up (increasing) learning rate, starting from zero.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup: target learning rate is reached at warmup.
    """

    def __init__(self, optimizer, warmup, milestones, gamma=0.1, verbose=False, last_epoch=-1):
        self.warmup = warmup
        super(LinearWarmupScheduler, self).__init__(optimizer, milestones, gamma, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch >= self.warmup:
            return super().get_lr()
        return [base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs]
