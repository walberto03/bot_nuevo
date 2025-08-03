# trading_bot/src/utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss para clasificación binaria:
    FL(p_t) = −α * (1 − p_t)^γ * log(p_t)
    donde p_t es la probabilidad estimada para la clase verdadera.
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 1.0, reduction: str = "mean", pos_weight=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        # Puedes pasar pos_weight como en BCEWithLogitsLoss si lo necesitas
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        logits: raw output sin sigmoid, shape (N,1)
        targets: 0/1 float tensor, shape (N,1)
        """
        # BCE con logits (aplica sigmoid internamente) pero sin reduction
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets,
            weight=self.pos_weight,
            reduction="none"
        )
        # probabilidad predicha
        p_t = torch.exp(-bce_loss)
        # factor focal
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss