import logging

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CFMToyModel(nn.Module):
    """CFM onto circle with learned pairwise interactions"""

    def __init__(self):
        super().__init__()
        # Base velocity field network
        self.base_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        # Learned interaction term
        self.interaction_net = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        batch_size = x.size(0)
        positions = x[:, :2]
        times = x[:, -1]
        base_velocity = self.base_net(x)

        # Compute pairwise features
        # Compute relative displacement
        # (B, 1, 2) - (1, B, 2) -> (B, B, 2)
        dx = positions.unsqueeze(1) - positions.unsqueeze(0)
        # Compute distances
        # (B, B, 1)
        distances = torch.norm(dx, dim=-1, keepdim=True)
        # Normalised displacrments
        # (B, B, 2)
        dx_norm = dx / (distances + 1e-6)
        # Pairwise features + time
        # (B, B, 4)
        interaction_features = torch.cat(
            [
                dx_norm,
                distances,
                torch.ones(distances.size(), device=x.device),
                times.unsqueeze(1).expand(batch_size, -1, -1),
            ],
            dim=-1,
        ).view(batch_size * batch_size, -1)
        interaction_forces = self.interaction_net(interaction_features).view(
            batch_size, batch_size, 2
        )
        # Mask diagonal and average forces on each particle
        mask = ~torch.eye(batch_size, device=x.device, dtype=bool)
        masked_interaction_forces = (interaction_forces * mask.unsqueeze(-1)).sum(
            dim=1
        ) / (batch_size - 1)
        return base_velocity + masked_interaction_forces
