import torch
import torch.nn as nn


class Trainer(object):
    def __init__(self, model, criterion, optimizer_scheduler, learning_rate, model_saver, checkpoint, logger,
                 frozen_bn=True):

        self.model = model
        self.criterion = criterion
        self.optimizer_scheduler = optimizer_scheduler
        self.learning_rate = learning_rate
        self.model_saver = model_saver
        self.checkpoint = checkpoint
        self.logger = logger
        self.frozen_bn = frozen_bn

        if torch.cuda.is_available():
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model_saver.dir_path = self.logger.save_dir(self.model_saver.dir_path)

        self.current_lr = 0
        self.total = 0

        self.model = nn.DataParallel(self.model).to(self.device)
        self.criterion = nn.DataParallel(self.criterion).to(self.device)
        self.model.train()

        if self.frozen_bn:
            def fix_bn(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.model.apply(fix_bn)

    def update_learning_rate(self, optimizer, epoch, step):
        lr = self.learning_rate.get_learning_rate(epoch, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        self.current_lr = lr

    def train(self, data_loader, epochs=50):
        self.logger.report_time("Start")
        self.steps = 0

        if hasattr(self.learning_rate, "epochs"):
            self.learning_rate.epochs = epochs

        self.checkpoint.restore_model(self.model, self.device, self.logger)

        epoch, iter_delta = self.checkpoint.restore_counter()
        self.steps = epoch * self.total + iter_delta

        optimizer = self.optimizer_scheduler.create_optimizer(self.model.parameters(), self.learning_rate)

        self.logger.report_time("Init")

        self.model.train()

        if self.frozen_bn:
            def fix_bn(m):
                classname = m.__class__.__name__
                if classname.find("BatchNorm") != -1:
                    m.eval()

            self.model.apply(fix_bn)

        while True:
            self.logger.info("Training epoch " + str(epoch))
            self.logger.epoch(epoch)
            self.total = len(data_loader)

            # torch.cuda.empty_cache()

            i = 0
            for batch in data_loader:
                batch["image"] = batch["image"].float()
                self.update_learning_rate(optimizer, epoch, self.steps)
                self.logger.report_time("Data loading")

                if self.logger.verbose:
                    torch.cuda.synchronize(self.device)

                self.train_step(self.model, self.criterion, optimizer, batch, epoch=epoch, step=self.steps)

                if self.logger.verbose:
                    torch.cuda.synchronize(self.device)
                self.logger.report_time("Forwarding ")

                self.model_saver.maybe_save_model(self.model, epoch, self.steps, self.logger)

                self.steps += 1
                self.logger.report_eta(self.steps, self.total, epoch)
            epoch += 1
            if epoch > epochs:
                self.model_saver.save_checkpoint(self.model, "final")
                self.logger.info("Training done")
                break

    def train_step(self, model, criterion, optimizer, batch, epoch, step):
        optimizer.zero_grad()

        image = batch["image"]
        pred = model.forward(image)

        l, metrics = criterion(pred, batch)

        if isinstance(l, dict):
            line = []
            loss = torch.tensor(0.).cuda()
            for key, l_val in l.items():
                loss += l_val.mean()
                line.append("loss_{0}:{1:.4f}".format(key, l_val.mean()))
        else:
            loss = l.mean()
        loss.backward()
        optimizer.step()

        if step % self.logger.log_interval == 0:
            if isinstance(l, dict):
                line = "\t".join(line)
                log_info = "\t".join(
                    ["step:{:6d}", "epoch:{:3d}", "{}", "lr:{:.4f}"]
                ).format(step, epoch, line, self.current_lr)
                self.logger.info(log_info)
            else:
                self.logger.info(
                    "step: %6d, epoch: %3d, loss: %.6f, lr: %f" % (step, epoch, loss.item(), self.current_lr)
                )
            self.logger.add_scalar("loss", loss, step)
            self.logger.add_scalar("learning_rate", self.current_lr, step)
            for name, metric in metrics.items():
                self.logger.add_scalar(name, metric.mean(), step)
                self.logger.info("%s: %6f" % (name, metric.mean()))

            self.logger.report_time("Logging")

    def validate(self, data_loader):
        pass

    def validate_step(self, data_loader, ):
        pass
