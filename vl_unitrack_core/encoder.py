"""Channel-fused joint encoding using supplied Fast-iTPN components."""

import torch
from torch import nn


def pair_views(uav, ground):
    """Pair preprocessed RGB tensors, UAV first, along the channel axis."""
    if uav.shape != ground.shape or uav.ndim != 4 or uav.shape[1] != 3:
        raise ValueError("Expected two matching (B, 3, H, W) tensors.")
    return torch.cat([uav, ground], dim=1)


class UnifiedJointEncoder(nn.Module):
    """Expose the joint-input adaptation, not a second backbone implementation.

    ``backbone`` supplies patch_embed, bridge/main blocks, position/type
    embeddings and normalization from Fast-iTPN. Its stem accepts six channels.
    Paired images are fused before tokenization, not concatenated as two
    independently tokenized view sequences.
    """

    def __init__(self, backbone):
        super().__init__()
        if backbone.patch_embed.proj.weight.shape[1] != 6:
            raise ValueError("The joint patch stem must accept six channels.")
        self.backbone = backbone

    @staticmethod
    def _mask(image, boxes):
        height, width = image.shape[-2:]
        x0 = (boxes[:, 0] * width).unsqueeze(1)
        y0 = (boxes[:, 1] * height).unsqueeze(1)
        w = (boxes[:, 2] * width).unsqueeze(1)
        h = (boxes[:, 3] * height).unsqueeze(1)
        xs = torch.arange(width, device=image.device)
        ys = torch.arange(height, device=image.device)
        xm = ((xs >= x0) & (xs < x0 + w)).float()
        ym = ((ys >= y0) & (ys < y0 + h)).float()
        return xm.unsqueeze(1) * ym.unsqueeze(2)

    def _template_indication(self, image, boxes):
        body = self.backbone
        mask = self._mask(image, boxes)
        mask = mask.unfold(1, body.patch_size, body.patch_size)
        mask = mask.unfold(2, body.patch_size, body.patch_size)
        mask = mask.mean(dim=(3, 4)).flatten(1)
        fg = body.template_foreground_token[None, None].expand(
            mask.size(0), mask.size(1), body.embed_dim
        )
        bg = body.template_background_token[None, None].expand_as(fg)
        return fg * mask.unsqueeze(-1) + bg * (1 - mask.unsqueeze(-1))

    def forward(self, templates, searches, template_boxes_uav, template_boxes_gro):
        """Return (B, L, C) tokens, search tokens first.

        Images are lists of (B, 6, H, W) tensors. Template boxes are lists of
        normalized xywh tensors (B, 4), describing the visual templates.
        One box per view may be supplied when all templates share that box.
        No search-frame ground truth enters this operation.
        """
        if not templates or not searches:
            raise ValueError("At least one template and search tensor are required.")
        if any(t.ndim != 4 or t.shape[1] != 6 for t in [*templates, *searches]):
            raise ValueError("Images must have shape (B, 6, H, W).")
        body = self.backbone
        batch = searches[0].shape[0]
        nt, ns = len(templates), len(searches)
        z = torch.stack(templates, dim=1).flatten(0, 1)
        x = torch.stack(searches, dim=1).flatten(0, 1)
        if body.token_type_indicate:
            indicators = []
            for boxes in (template_boxes_uav, template_boxes_gro):
                if len(boxes) == 1:
                    boxes = list(boxes) * nt
                if len(boxes) != nt:
                    raise ValueError("Template boxes must match the template count.")
                boxes = torch.stack(boxes, dim=1).flatten(0, 1)
                indicators.append(self._template_indication(z, boxes))

        z, x = body.patch_embed(z), body.patch_embed(x)
        if not body.convmlp and body.stop_grad_conv1:
            x = x.detach() * 0.9 + x * 0.1
        for block in body.blocks[:-body.num_main_blocks]:
            z, x = block(z), block(x)
        x = x.flatten(2).transpose(1, 2)
        z = z.flatten(2).transpose(1, 2)
        if body.pos_embed is not None:
            x = x + body.pos_embed[:, :body.num_patches_search, :]
            z = z + body.pos_embed[:, body.num_patches_search:, :]
        if body.token_type_indicate:
            x = x + body.search_token[None, None].expand_as(x)
            z = z + 0.5 * indicators[0] + 0.5 * indicators[1]
        x = x.reshape(batch, ns * x.size(1), x.size(2))
        z = z.reshape(batch, nt * z.size(1), z.size(2))
        tokens = torch.cat([x, z], dim=1)
        if body.cls_token is not None:
            tokens = torch.cat([body.cls_token.expand(batch, -1, -1), tokens], dim=1)
        tokens = body.pos_drop(tokens)
        bias = body.rel_pos_bias() if body.rel_pos_bias is not None else None
        for block in body.blocks[-body.num_main_blocks:]:
            tokens = block(tokens, bias)
        tokens = body.norm(tokens)
        return body.fc_norm(tokens) if body.fc_norm is not None else tokens
