"""Confidence-switched, target-pooled cross-view cosine distillation."""

import torch
from torch.nn import functional as F


def confidence_modulated_distillation(features, scores_uav, scores_gro,
                                     target_map_uav, target_map_gro,
                                     margin=0.05):
    """Compute CMD, without any outer tracking-loss weighting.

    Features: uav_final/gro_final (B,C,H,W).
    Scores: already-sigmoid classification responses (B,1,H,W).
    Target maps: Gaussian target-region weights (B,1,H,W), learning only.
    Each active direction is averaged separately, then the two means are added.
    """
    vu = (features["uav_final"] * target_map_uav).sum(dim=(2, 3))
    vg = (features["gro_final"] * target_map_gro).sum(dim=(2, 3))
    vu = vu / (target_map_uav.sum(dim=(2, 3)) + 1e-6)
    vg = vg / (target_map_gro.sum(dim=(2, 3)) + 1e-6)
    cu = scores_uav.detach().flatten(1).max(dim=1)[0]
    cg = scores_gro.detach().flatten(1).max(dim=1)[0]
    loss = torch.tensor(0.0, device=vu.device)
    for mask, teacher, student in (
        (cu > cg + margin, vu, vg),
        (cg > cu + margin, vg, vu),
    ):
        selected = torch.where(mask)[0]
        if len(selected) > 0:
            target = torch.ones(len(selected), device=vu.device)
            loss = loss + F.cosine_embedding_loss(
                student[selected], teacher[selected].detach(), target
            )
    return loss
