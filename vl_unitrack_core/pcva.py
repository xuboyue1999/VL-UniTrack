"""Prompt-conditioned cross-view adaptation of a common joint feature map."""

import torch
from torch import nn


class CrossViewAdapter(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=True, dropout=0.1):
        super().__init__()
        if dim % num_heads:
            raise ValueError("Feature dimension must be divisible by num_heads.")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim * 2, dim), nn.Dropout(dropout),
        )
        for parameter in self.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)

    def forward(self, x_q, x_kv, prompt_token=None):
        batch, channels, height, width = x_q.shape
        n = height * width
        q = x_q.flatten(2).transpose(1, 2)
        kv = x_kv.flatten(2).transpose(1, 2)
        if prompt_token is not None:
            kv = torch.cat([prompt_token, kv], dim=1)
        q_norm, kv_norm = self.norm_q(q), self.norm_kv(kv)
        qh = self.q_proj(q_norm).reshape(
            batch, n, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        kh = self.k_proj(kv_norm).reshape(
            batch, -1, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        vh = self.v_proj(kv_norm).reshape(
            batch, -1, self.num_heads, channels // self.num_heads
        ).permute(0, 2, 1, 3)
        attention = ((qh @ kh.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        aligned = (self.attn_drop(attention) @ vh).transpose(1, 2)
        aligned = aligned.reshape(batch, n, channels)
        x = q + self.proj_drop(self.proj(aligned))
        x = x + self.ffn(self.norm_ffn(x))
        return x.transpose(1, 2).reshape(batch, channels, height, width)


class PromptGuidedCrossViewAdapter(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.view_token_uav = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.view_token_gro = nn.Parameter(torch.zeros(1, dim, 1, 1))
        nn.init.trunc_normal_(self.view_token_uav, std=0.02)
        nn.init.trunc_normal_(self.view_token_gro, std=0.02)
        self.fusion_u2g = CrossViewAdapter(dim, num_heads, dropout=dropout)
        self.fusion_g2u = CrossViewAdapter(dim, num_heads, dropout=dropout)
        self.gamma_u = nn.Parameter(torch.zeros(1))
        self.gamma_g = nn.Parameter(torch.zeros(1))

    def forward(self, joint_feature, view_prompts):
        base_u = joint_feature + self.view_token_uav
        base_g = joint_feature + self.view_token_gro
        pu, pg = view_prompts if view_prompts is not None else (None, None)
        # The querying view's prompt is prepended to the OTHER view's K/V.
        cross_u = self.fusion_u2g(base_u, base_g, pu)
        cross_g = self.fusion_g2u(base_g, base_u, pg)
        # Adapter outputs already contain their internal attention/FFN residuals.
        return {
            "uav_final": base_u + self.gamma_u * cross_u,
            "gro_final": base_g + self.gamma_g * cross_g,
        }
