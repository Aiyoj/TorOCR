import numpy as np

from addict import Dict


class ConstantLearningRate(object):

    def __init__(self, configs):
        self.cfg = Dict(configs)

        self.lr = self.cfg.get("lr", 0.0001)

    def get_learning_rate(self, epoch, step):
        return self.lr


class DecayLearningRate(object):

    def __init__(self, configs):
        self.cfg = Dict(configs)

        self.lr = self.cfg.get("lr", 0.007)
        self.epochs = self.cfg.get("epochs", 200)
        self.factor = self.cfg.get("factor", 0.7)

    def get_learning_rate(self, epoch, step):
        rate = np.power(1.0 - epoch / float(self.epochs + 1), self.factor)
        return rate * self.lr


class CosineDecayWithWarmupLearningRate(object):
    def __init__(self, configs):
        self.cfg = Dict(configs)

        self.lr = self.cfg.get("lr", 0.001)
        self.epochs = self.cfg.get("epochs", 100)
        self.step_each_epoch = self.cfg.get("step_each_epoch", 47)
        self.warmup_step = self.cfg.get("warmup_step", 47)

    def get_learning_rate(self, epoch, step):
        if step < self.warmup_step:
            return self.lr * (1 * step / self.warmup_step)

        return self.lr * (np.cos((step - self.warmup_step) * (np.pi / (self.epochs * self.step_each_epoch))) + 1) / 2
