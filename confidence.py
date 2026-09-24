import torch
import torch.nn as nn
import torch.nn.functional as F

class ConfidenceLens(nn.Module):

    def __init__(
        self,
        dim,
        hidden_dim=None,
        tau=0.4,
        lambda_=1.0,
        epsilon=1e-6,
        dropout=0.0,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim or dim)

        self.dim = int(dim)
        self.tau = float(tau)
        self.lambda_ = float(lambda_)
        self.epsilon = float(epsilon)

        # h = epsilon(f).
        self.focus_mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim),
        )

        self.variance_mlp = nn.Sequential(
            nn.Linear(self.dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.dim),
        )

        # update direction and element-wise update magnitude.
        self.delta_projection = nn.Linear(self.dim, self.dim)
        self.gate_projection = nn.Linear(self.dim, self.dim)


        self.focal_projection_s = nn.Linear(self.dim, hidden_dim)
        self.focal_projection_t = nn.Linear(hidden_dim, 1)

    def forward(self, feature):
        if feature.ndim != 2:
            raise ValueError(
                "ConfidenceLens expects a sample-level tensor [B, D], "
                f"but received shape {tuple(feature.shape)}."
            )
        if feature.size(-1) != self.dim:
            raise ValueError(
                f"ConfidenceLens was built for dim={self.dim}, "
                f"but received dim={feature.size(-1)}."
            )


        hidden = feature + self.focus_mlp(feature)


        variance = F.softplus(self.variance_mlp(hidden))


        delta_variance = self.delta_projection(variance)
        variance_gate = torch.sigmoid(self.gate_projection(variance))
        corrected_variance = variance + variance_gate * delta_variance
        corrected_variance = corrected_variance.clamp_min(self.epsilon)


        unreliability = torch.log1p(
            corrected_variance.mean(dim=-1, keepdim=True)
        )


        focal_hidden = F.gelu(self.focal_projection_s(hidden))
        focal = self.tau + self.lambda_ * F.gelu(
            self.focal_projection_t(focal_hidden)
        )
        focal = focal.clamp_min(self.epsilon)


        confidence = torch.exp(-focal * unreliability).clamp(max=1.0)

        details = {
            "hidden": hidden,
            "variance": variance,
            "corrected_variance": corrected_variance,
            "unreliability": unreliability,
            "focal": focal,
        }
        return confidence, details
