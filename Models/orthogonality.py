import torch.nn.functional as F

def orth_loss(private, common, eps=1e-8):
    """Sample-wise soft orthogonality loss."""
    private = private.flatten(1)
    common = common.flatten(1)

    cosine = F.cosine_similarity(
        private,
        common,
        dim=1,
        eps=eps,
    )

    return cosine.abs().mean()
