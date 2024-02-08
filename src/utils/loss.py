from torch import nn


class CustomGaussianNLLLoss(nn.Module):
    def __init__(self, reduction: str = "mean", full: bool = False, eps: float = 1e-6):
        super().__init__()
        self.reduction = reduction
        self.full = full
        self.eps = eps

    def forward(self, input, target, std, var_coeff):
        return (
            nn.functional.gaussian_nll_loss(
                input,
                target,
                std,
                full=self.full,
                eps=self.eps,
                reduction=self.reduction,
            )
            + var_coeff * (std**2).sum()
        )  # Untested yet but should work
