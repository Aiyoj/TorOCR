import torch
import torch.nn as nn


class CTCLoss(nn.Module):
    def __init__(self, blank_idx, reduction="mean"):
        super(CTCLoss, self).__init__()
        self.loss_func = torch.nn.CTCLoss(blank=blank_idx, reduction=reduction, zero_infinity=True)

    def forward(self, pred, batch):
        batch_size = pred.size(0)
        label, label_length = batch["targets"], batch["targets_lengths"]
        pred = pred.log_softmax(2)
        pred = pred.permute(1, 0, 2)
        preds_lengths = torch.tensor([pred.size(0)] * batch_size, dtype=torch.long)
        loss = self.loss_func(pred, label, preds_lengths, label_length)
        metrics = dict(ctc_loss=loss)

        return loss, metrics
