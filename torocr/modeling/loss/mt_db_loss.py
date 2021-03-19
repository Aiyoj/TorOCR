import torch.nn as nn
from collections import OrderedDict


class DiceLoss(nn.Module):
    """
    DiceLoss on binary.
    For SegDetector without adaptive module.
    """

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        from torocr.modeling.loss.dice_loss import DiceLoss as Loss
        self.loss = Loss(eps)

    def forward(self, pred, batch):
        loss = self.loss(pred["binary"], batch["gt"], batch["mask"])
        return loss, dict(dice_loss=loss)


class L1BalanceCELoss(nn.Module):
    """
    Balanced CrossEntropy Loss on `binary`,
    MaskL1Loss on `thresh`,
    DiceLoss on `thresh_binary`.
    Note: The meaning of inputs can be figured out in `SegDetectorLossBuilder`.
    """

    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, num_classes=1):
        super(L1BalanceCELoss, self).__init__()
        from torocr.modeling.loss.dice_loss import DiceLoss
        from torocr.modeling.loss.l1_loss import MaskL1Loss
        from torocr.modeling.loss.balance_cross_entropy_loss import BalanceCrossEntropyLoss

        self.num_classes = num_classes
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss()
        self.bce_loss = BalanceCrossEntropyLoss()

        self.l1_scale = l1_scale
        self.bce_scale = bce_scale

    def forward(self, pred, batch):
        if self.num_classes == 1:
            bce_loss = self.bce_loss(pred["binary"], batch["gt"], batch["mask"])
            metrics = dict(bce_loss=bce_loss)
            if "thresh" in pred:
                l1_loss, l1_metric = self.l1_loss(pred["thresh"], batch["thresh_map"], batch["thresh_mask"])
                dice_loss = self.dice_loss(pred["thresh_binary"], batch["gt"], batch["mask"])
                metrics["thresh_loss"] = dice_loss
                loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
                metrics.update(**l1_metric)
            else:
                loss = bce_loss
            return loss, metrics
        else:
            total_loss = 0
            metrics = OrderedDict()
            for i in range(self.num_classes):
                # shrink map loss
                bce_loss = self.bce_loss(pred["binary_{}".format(i)], batch["gt_{}".format(i)],
                                         batch["mask_{}".format(i)])
                metrics.update({"bce_loss_{}".format(i): bce_loss})

                # thresh map loss
                l1_loss, l1_metric = self.l1_loss(
                    pred["thresh_{}".format(i)], batch["thresh_map_{}".format(i)],
                    batch["thresh_mask_{}".format(i)]
                )

                # binary loss
                dice_loss = self.dice_loss(
                    pred["thresh_binary_{}".format(i)], batch["gt_{}".format(i)],
                    batch["mask_{}".format(i)]
                )
                metrics["thresh_loss_{}".format(i)] = dice_loss
                loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
                metrics.update({"l1_loss_{}".format(i): l1_metric["l1_loss"], "loss_{}".format(i): loss})

                total_loss += loss

            metrics.update({"total_loss": total_loss})

            return total_loss, metrics
