# src/utils/losses.py
import torch
import torch.nn.functional as F

def masked_bce_loss(logits, labels):
    """
    logits: (batch, num_tasks)
    labels: (batch, num_tasks) with NaN for missing
    """
    mask = ~torch.isnan(labels)
    if mask.sum() == 0:
        return logits.sum() * 0.0
    labels0 = torch.nan_to_num(labels, 0.0)
    bce = F.binary_cross_entropy_with_logits(logits, labels0, reduction="none")
    bce = bce * mask.float()
    return bce.sum() / mask.sum()
