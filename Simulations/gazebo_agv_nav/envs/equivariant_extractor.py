"""Arm 3: E(2)-steerable equivariant policy network (escnn), D4 group --
matches the D4 group used for arm 2's data augmentation and the G3
generalization benchmark (envs/symmetry.py), so all three arms are compared
under the same symmetry assumption.

The occupancy-grid observation goes through a D4-equivariant CNN stack
(R2Conv layers over regular-representation fields) and is pooled down to a
D4-*invariant* feature vector via GroupPooling. That invariant vector is
concatenated with the raw goal_relative vector and fed to SB3's normal MLP
head. Pooling to invariant (rather than carrying a steerable/regular output
all the way to the action heads) is a deliberate scope cut: full end-to-end
action-equivariance would also require the action distribution itself to
transform correctly (e.g. angular velocity flips sign under mirror), which
SB3's policy heads don't support out of the box. Baking the symmetry into
the *feature extractor* -- so the network structurally cannot tell a map and
its D4 transform apart -- is what's being tested against arm 1 (nothing) and
arm 2 (data augmentation) here.
"""
from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import escnn.nn as enn
from escnn.gspaces import flipRot2dOnR2


class D4EquivariantGridEncoder(nn.Module):
    """D4-equivariant CNN over the occupancy grid, pooled to an invariant vector."""

    def __init__(self, grid_size: int, out_dim: int = 64):
        super().__init__()
        self.gspace = flipRot2dOnR2(N=4)  # D4: 4 rotations x {identity, mirror}
        r_in = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        self.input_type = r_in

        r1 = enn.FieldType(self.gspace, 8 * [self.gspace.regular_repr])
        r2 = enn.FieldType(self.gspace, 16 * [self.gspace.regular_repr])
        r3 = enn.FieldType(self.gspace, 32 * [self.gspace.regular_repr])

        self.block1 = enn.SequentialModule(
            enn.R2Conv(r_in, r1, kernel_size=5, padding=2),
            enn.InnerBatchNorm(r1),
            enn.ReLU(r1, inplace=True),
            enn.PointwiseMaxPool(r1, kernel_size=2),
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(r1, r2, kernel_size=3, padding=1),
            enn.InnerBatchNorm(r2),
            enn.ReLU(r2, inplace=True),
            enn.PointwiseMaxPool(r2, kernel_size=2),
        )
        self.block3 = enn.SequentialModule(
            enn.R2Conv(r2, r3, kernel_size=3, padding=1),
            enn.InnerBatchNorm(r3),
            enn.ReLU(r3, inplace=True),
        )
        self.group_pool = enn.GroupPooling(r3)  # -> D4-invariant scalar fields

        pooled_hw = grid_size // 4
        n_invariant_channels = r3.size // self.gspace.fibergroup.order()
        self.head = nn.Linear(n_invariant_channels * pooled_hw * pooled_hw, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.group_pool(x)
        x = x.tensor.flatten(1)
        return torch.relu(self.head(x))


class D4EquivariantExtractor(BaseFeaturesExtractor):
    """Combined extractor for the Dict obs space: D4-equivariant CNN on
    'occupancy', tiny MLP on 'goal_relative', concatenated -- drop-in
    replacement for SB3's default CombinedExtractor via policy_kwargs."""

    def __init__(self, observation_space: gym.spaces.Dict, grid_feat_dim: int = 64):
        goal_dim = observation_space["goal_relative"].shape[0]
        super().__init__(observation_space, features_dim=grid_feat_dim + 32)
        grid_size = observation_space["occupancy"].shape[0]
        self.grid_encoder = D4EquivariantGridEncoder(grid_size, out_dim=grid_feat_dim)
        self.goal_mlp = nn.Sequential(nn.Linear(goal_dim, 32), nn.ReLU())

    def forward(self, observations: dict) -> torch.Tensor:
        occ = observations["occupancy"].float().unsqueeze(1)  # (B,1,H,W)
        grid_feat = self.grid_encoder(occ)
        goal_feat = self.goal_mlp(observations["goal_relative"].float())
        return torch.cat([grid_feat, goal_feat], dim=1)
