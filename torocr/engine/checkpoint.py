import os
import torch
import torch.nn as nn

from addict import Dict


class Checkpoint(object):
    def __init__(self, configs):

        self.cfg = Dict(configs)

        self.start_epoch = self.cfg.get("start_epoch", 0)
        self.start_iter = self.cfg.get("start_iter", 0)
        self.resume = self.cfg.get("resume", None)

    def restore_model(self, model, device, logger):
        if self.resume is None:
            return

        if not os.path.exists(self.resume):
            logger.warning("Checkpoint not found: " + self.resume)
            return

        logger.info("Resuming from " + self.resume)
        state_dict = torch.load(self.resume, map_location=device)
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)
        logger.info("Resumed from " + self.resume)

    def restore_counter(self):
        return self.start_epoch, self.start_iter
