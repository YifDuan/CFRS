import torch.nn.functional as F


def c_loss(estimated, target, alpha, beta):
    """Feature alignment loss: MSE plus cosine distance."""
    mse = F.mse_loss(estimated, target)
    cosine = F.cosine_similarity(estimated, target, dim=-1, eps=1e-8)
    cosine_loss = (1.0 - cosine).mean()
    return alpha * mse + beta * cosine_loss
