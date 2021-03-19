import torch

from addict import Dict


class OptimizerScheduler(object):

    def __init__(self, configs):
        self.cfg = Dict(configs)

        self.optimizer = self.cfg.get("optimizer", "SGD")
        self.optimizer_args = self.cfg.get("optimizer_args", {})
        self.lr = self.cfg.get("lr", 0.001)

        self.optimizer_args["lr"] = self.lr

    def create_optimizer(self, parameters, learning_rate):
        optimizer = getattr(torch.optim, self.optimizer)(
            parameters, **self.optimizer_args)
        if hasattr(learning_rate, "prepare"):
            learning_rate.prepare(optimizer)
        return optimizer
