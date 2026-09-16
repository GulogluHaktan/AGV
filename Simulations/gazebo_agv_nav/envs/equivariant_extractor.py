"""Arm 3: E(2)-steerable equivariant policy features (escnn), D4 group --
matches the group used for arm 2's augmentation and the G3 benchmark
(envs/symmetry.py), so all three arms rest on the same symmetry assumption.

What this computes, and why it is not simply "invariant features"
-----------------------------------------------------------------
The occupancy grid goes through a D4-equivariant CNN stack, whose final layer
emits VECTOR fields (the 2-D irrep of D4) alongside invariant scalar fields.
Averaging over space keeps vector fields equivariant, so after pooling we hold
a handful of world-frame vectors -- "clutter increases in this direction".

Those vectors are then read out against two world-frame reference directions
recovered from the observation itself: the robot heading h, and its
perpendicular h_perp. Because a vector and the reference rotate together, the
inner products are exactly rotation-invariant; and because a reflection
reverses orientation, h_perp picks up a sign, so v.h_perp is mirror-ODD while
v.h is mirror-even.

That asymmetry is deliberate. Fully D4-invariant features would be wrong here:
under a mirror the correct action is (linear, -angular), so a policy whose
features were mirror-invariant could not produce the flipped behaviour. The
feature set instead transforms exactly like the action does -- invariant under
rotation, left/right-odd under reflection -- which is also how `goal_body`
behaves. Rotation-invariance is what the G3 benchmark needs: `d4_orbit` drops
the degenerate mirror for the symmetric layouts, so the held-out set is
rotations.

Features are laid out as [even..., odd...] so `scripts/test_equivariance.py`
can check the sign pattern rather than take the claim on trust.

The previous version of this module pooled the group channels with
GroupPooling and then flattened the remaining spatial map into a Linear layer.
That silently destroyed the property the arm exists for: GroupPooling makes
features invariant to the group CHANNEL action only, while the spatial map
still transforms, so the flatten saw a rotated layout and produced a different
answer -- measured at ~25% relative error, once misattributed to escnn's kernel
discretization. escnn is exact to machine precision here (HANDOFF §15).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import escnn.nn as enn
from escnn.gspaces import flipRot2dOnR2

# escnn's vector-field basis and our world frame disagree by a swap of the two
# components: escnn's rotations come out as the inverse of `transform_direction`'s
# and its reflection axis is the other one. Conjugating by this swap reconciles
# both at once -- identified empirically by least-squares fitting the pooled
# field's transformation matrix for all eight group elements and comparing it
# with `transform_direction`, rather than by reasoning about conventions.
_SWAP_XY = True


class D4EquivariantGridEncoder(nn.Module):
    """D4-equivariant CNN over the occupancy grid.

    Returns (vectors, invariants): `vectors` is (B, n_vec, 2) of world-frame
    vector fields, `invariants` is (B, n_inv) of D4-invariant scalars, both
    pooled over space. Spatial mean-pooling is itself equivariant (the mean of
    rotated vectors is the rotated mean), which is what lets the readout stay
    exact.
    """

    def __init__(self, n_vec: int = 16, n_inv: int = 32, width: int = 8):
        super().__init__()
        self.gspace = flipRot2dOnR2(N=4)  # D4: 4 rotations x {identity, mirror}
        self.n_vec, self.n_inv = n_vec, n_inv

        t_in = enn.FieldType(self.gspace, [self.gspace.trivial_repr])
        t1 = enn.FieldType(self.gspace, width * [self.gspace.regular_repr])
        t2 = enn.FieldType(self.gspace, (2 * width) * [self.gspace.regular_repr])
        # final layer emits vector fields + invariant scalar fields side by side
        t_out = enn.FieldType(
            self.gspace,
            n_vec * [self.gspace.irrep(1, 1)] + n_inv * [self.gspace.trivial_repr])
        self.input_type = t_in

        self.block1 = enn.SequentialModule(
            enn.R2Conv(t_in, t1, kernel_size=5, padding=2),
            enn.InnerBatchNorm(t1),
            enn.ReLU(t1, inplace=True),
            enn.PointwiseMaxPool(t1, kernel_size=2),
        )
        self.block2 = enn.SequentialModule(
            enn.R2Conv(t1, t2, kernel_size=3, padding=1),
            enn.InnerBatchNorm(t2),
            enn.ReLU(t2, inplace=True),
            enn.PointwiseMaxPool(t2, kernel_size=2),
        )
        # no nonlinearity after the vector-field layer: a pointwise ReLU would
        # not be equivariant on irrep fields
        self.to_fields = enn.R2Conv(t2, t_out, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        y = self.to_fields(self.block2(self.block1(
            enn.GeometricTensor(x, self.input_type)))).tensor
        y = y.mean(dim=(2, 3))                       # pool over space
        vec = y[:, : 2 * self.n_vec].reshape(-1, self.n_vec, 2)
        inv = y[:, 2 * self.n_vec:]
        if _SWAP_XY:
            vec = vec.flip(-1)
        return vec, inv


class D4EquivariantExtractor(BaseFeaturesExtractor):
    """SB3 features extractor for the Dict observation space.

    Output layout is [even | odd] before the MLP: the odd half flips sign under
    a reflection, matching how the angular action and `goal_body`'s left/right
    component behave. See the module docstring.
    """

    def __init__(self, observation_space: gym.spaces.Dict, grid_feat_dim: int = 64,
                 n_vec: int = 16, n_inv: int = 32):
        super().__init__(observation_space, features_dim=grid_feat_dim)
        self.encoder = D4EquivariantGridEncoder(n_vec=n_vec, n_inv=n_inv)
        self.n_vec = n_vec
        # even: v.h (n_vec), invariants (n_inv), goal forward, |goal|
        # odd : v.h_perp (n_vec), goal left
        self.n_even = n_vec + n_inv + 2
        self.n_odd = n_vec + 1
        self.mlp = nn.Sequential(
            nn.Linear(self.n_even + self.n_odd, grid_feat_dim), nn.ReLU())

    def raw_features(self, observations: dict) -> torch.Tensor:
        """The [even | odd] vector fed to the MLP. Exposed so the equivariance
        test can check the sign pattern directly, before the MLP mixes it."""
        occ = observations["occupancy"].float().unsqueeze(1)   # (B,1,H,W)
        vec, inv = self.encoder(occ)

        heading = observations["heading"].float()              # (B,2) world frame
        goal_body = observations["goal_body"].float()          # (B,2) robot frame
        c, s = heading[:, 0], heading[:, 1]
        # world-frame goal offset: R(yaw) @ goal_body
        gx = c * goal_body[:, 0] - s * goal_body[:, 1]
        gy = s * goal_body[:, 0] + c * goal_body[:, 1]
        goal_w = torch.stack([gx, gy], dim=1)
        goal_norm = torch.linalg.norm(goal_w, dim=1, keepdim=True)

        h = heading
        h_perp = torch.stack([-h[:, 1], h[:, 0]], dim=1)       # rotate h by +90
        along = torch.einsum("bnd,bd->bn", vec, h)             # even
        across = torch.einsum("bnd,bd->bn", vec, h_perp)       # odd

        even = torch.cat([along, inv, goal_body[:, :1], goal_norm], dim=1)
        odd = torch.cat([across, goal_body[:, 1:2]], dim=1)
        return torch.cat([even, odd], dim=1)

    def forward(self, observations: dict) -> torch.Tensor:
        return self.mlp(self.raw_features(observations))
