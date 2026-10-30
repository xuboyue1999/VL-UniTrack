"""Frozen CLIP priors, learnable visual fusion, and sequence-local caching."""

import torch
from torch import nn
from torch.nn import functional as F


class ViewpointPromptEncoder(nn.Module):
    def __init__(self, clip_model, tokenize, out_channels, *, uav_texts, ground_texts):
        """Fill in the UAV and ground prompt descriptions via these arguments.

        No preset descriptions are bundled. Supply a nonempty list or tuple of
        strings for each view before constructing the module.
        """
        super().__init__()
        for phrases in (uav_texts, ground_texts):
            if (
                not isinstance(phrases, (list, tuple))
                or not phrases
                or any(not isinstance(text, str) or not text.strip() for text in phrases)
            ):
                raise ValueError("Fill in a nonempty list of prompt strings for each view.")
        self.clip_model = clip_model.requires_grad_(False).eval()
        device = next(clip_model.parameters()).device
        with torch.no_grad():
            for name, phrases in (
                ("text_uav", uav_texts), ("text_gro", ground_texts)
            ):
                text = clip_model.encode_text(tokenize(phrases).to(device)).float()
                text = text.mean(dim=0, keepdim=True)
                self.register_buffer(name, text / text.norm(dim=-1, keepdim=True))
        self.prompt_proj = nn.Linear(clip_model.text_projection.size(1), out_channels)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        nn.init.zeros_(self.prompt_proj.weight)
        nn.init.zeros_(self.prompt_proj.bias)
        self.register_buffer("_cached_uav", None, persistent=False)
        self.register_buffer("_cached_gro", None, persistent=False)
        self._cache_stamp = None

    def forward(self, initial_template):
        """Uncached, differentiable fusion; use this path for model learning.

        Input is the preprocessed paired tracking template (B, 6, H, W).
        Only resize it for CLIP here: do not apply a second normalization.
        CLIP features are frozen; alpha and the projection retain gradients.
        """
        if initial_template.ndim != 4 or initial_template.shape[1] != 6:
            raise ValueError("Expected a paired (B, 6, H, W) template.")
        uav = F.interpolate(initial_template[:, :3], (224, 224), mode="bicubic")
        gro = F.interpolate(initial_template[:, 3:], (224, 224), mode="bicubic")
        with torch.no_grad():
            vu = self.clip_model.encode_image(uav).float()
            vg = self.clip_model.encode_image(gro).float()
            vu = vu / vu.norm(dim=-1, keepdim=True)
            vg = vg / vg.norm(dim=-1, keepdim=True)
        # Alpha scales the visual term; this is NOT alpha*T + (1-alpha)*V.
        pu = self.prompt_proj(self.text_uav + self.alpha * vu)
        pg = self.prompt_proj(self.text_gro + self.alpha * vg)
        return pu.unsqueeze(1), pg.unsqueeze(1)

    def clear_cache(self):
        self._cached_uav = None
        self._cached_gro = None
        self._cache_stamp = None

    def _state_stamp(self):
        # Catch normal in-place edits as well as tensor replacement. Mutating
        # tensors via .data bypasses PyTorch versioning and is not supported.
        tensors = list(self.parameters()) + [
            value for name, value in self.named_buffers()
            if name not in ("_cached_uav", "_cached_gro")
        ]
        return tuple((id(value), value._version) for value in tensors)

    @torch.no_grad()
    def initialize_sequence(self, initial_template):
        """Call once for EACH new sequence, after loading parameters/text."""
        if self.training:
            raise RuntimeError("Sequence caching is only available in eval mode.")
        self.clear_cache()
        self._cached_uav, self._cached_gro = self(initial_template)
        self._cache_stamp = self._state_stamp()
        return self._cached_uav, self._cached_gro

    def cached_prompts(self):
        """Reuse the initial prompts; online template updates do not change them."""
        if self.training or self._cache_stamp is None:
            raise RuntimeError("Call eval() and initialize_sequence() first.")
        if self._cache_stamp != self._state_stamp():
            self.clear_cache()
            raise RuntimeError("Prompt state changed; initialize_sequence() again.")
        return self._cached_uav, self._cached_gro

    def train(self, mode=True):
        super().train(mode)
        self.clip_model.eval()
        if mode:
            self.clear_cache()
        return self

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        return super()._apply(fn, recurse=recurse)

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)
