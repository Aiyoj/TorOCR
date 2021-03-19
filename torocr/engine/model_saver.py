import os
import torch
import torch.nn as nn

from addict import Dict


class ModelSaver(object):
    def __init__(self, configs):
        self.cfg = Dict(configs)

        self.dir_path = self.cfg.get("dir_path", "model")
        self.save_interval = self.cfg.get("save_interval", 50)
        self.signal_path = self.cfg.get("signal_path", "save")

    def maybe_save_model(self, model, epoch, step, logger):
        if step % self.save_interval == 0:
            self.save_model(model, epoch, step)
            logger.report_time("Saving ")
            logger.iter(step)

    def save_model(self, model, epoch=None, step=None):
        if isinstance(model, dict):
            for name, net in model.items():
                checkpoint_name = self.make_checkpoint_name(name, epoch, step)
                self.save_checkpoint(net, checkpoint_name)
        else:
            checkpoint_name = self.make_checkpoint_name("model", epoch, step)
            self.save_checkpoint(model, checkpoint_name)

    def save_checkpoint(self, net, name):
        os.makedirs(self.dir_path, exist_ok=True)
        if isinstance(net, nn.DataParallel):
            torch.save(net.module.state_dict(), os.path.join(self.dir_path, name))
        else:
            torch.save(net.state_dict(), os.path.join(self.dir_path, name))

    @staticmethod
    def make_checkpoint_name(name, epoch=None, step=None):
        if epoch is None or step is None:
            checkpoint_name = "{}_latest".format(name)
        else:
            checkpoint_name = "{}_epoch_{}_minibatch_{}".format(name, epoch, step)

        return checkpoint_name
