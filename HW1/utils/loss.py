import torch
import torch.nn.functional as F


def ce_loss(logits, targets, reduction="none"):
    """
    Compute cross entropy loss that supports both one-hot encoded targets and
    class indices.

    When the shapes of logits and targets match, the targets are assumed to be
    one-hot encoded. Otherwise, targets are assumed to be class indices.

    Args:
        logits (torch.Tensor): Logit values with shape
            [batch_size, num_classes].
        targets (torch.Tensor): Targets with shape [batch_size] if using class
            indices, or [batch_size, num_classes] if one-hot encoded.
        reduction (str, optional): Specifies the reduction to apply to the
            output.
            Options are "none" or "mean". Default is "none".

    Returns:
        torch.Tensor: The computed cross entropy loss.
    """
    if logits.shape == targets.shape:
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == "none":
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits_u_str, logits_u_w, threshold=0.6):
    """
    Compute consistency regularization loss for semi-supervised learning with a
    fixed threshold.

    This loss encourages consistency between predictions of strongly and weakly
    augmented versions of the same unlabeled sample. Only samples with a
    maximum probability exceeding the threshold are considered.

    Args:
        logits_u_str (torch.Tensor): Logits from strongly augmented unlabeled
            samples.
        logits_u_w (torch.Tensor): Logits from weakly augmented unlabeled
            samples.
        threshold (float, optional): Confidence threshold for pseudo-labeling.
            Default is 0.6.

    Returns:
        torch.Tensor: The computed consistency loss.
    """
    pseudo_label = torch.softmax(logits_u_w, dim=1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(threshold).float()
    loss = (
        ce_loss(logits_u_str, targets_u, reduction="none") * mask
    ).mean()
    return loss
