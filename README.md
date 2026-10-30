# VL-UniTrack: A Unified Framework with Visual-Language Prompts for UAV-Ground Visual Tracking (Under review)

## Core Algorithm Reference

This is a partial source package for reviewing the core algorithms, covering joint visual encoding,
Visual-Language View Prompting (VLVP), Prompt-guided Cross-View Adaptation
(PCVA), and Confidence-Modulated Distillation (CMD). It is not a complete
tracker, a runnable demo, or a training/benchmark reproduction package.
The files present component-level computations; system integration is
outside the scope of this package.

## Source Map

| File | Role |
| --- | --- |
| `encoder.py` | Paired channel fusion, template-region indication, joint token assembly and backbone forwarding |
| `vlvp.py` | Frozen CLIP text/visual features, learnable fusion and projection, sequence-local prompt cache |
| `pcva.py` | Global view tokens, bidirectional prompt-conditioned attention, FFN and learned residual scaling |
| `cmd.py` | Confidence selection and stop-gradient cross-view distillation |

Files are in the `vl_unitrack_core` package. The implementation uses PyTorch.
The encoder and VLVP reference modules require external backbone and CLIP
components that are not supplied, downloaded or built here. Standard backbone
blocks, prediction heads and complete model assembly are omitted.
The package contains no tracking loop, image-loading or cropping pipeline,
online template-update or postprocessing implementation, optimization recipe,
experiment configuration, checkpoint, benchmark runner, or runtime-environment
setup. It cannot produce tracking results from a video on its own.

## Computation

**Joint encoder.** UAV and ground RGB tensors are concatenated along channels:
`pair_views(U, G)` has six channels. A shared six-channel patch stem jointly
embeds them. It produces one joint spatial token grid, not two concatenated
view-token grids. Search and template tokens are subsequently concatenated.
Template-region embeddings average the two views' foreground/background
indications. PCVA derives two view-conditioned maps from the common joint map.

**VLVP.** Viewpoint descriptions are supplied by the caller through the required
`uav_texts` and `ground_texts` arguments of `ViewpointPromptEncoder`; no preset
prompt strings are bundled. Text features are averaged within each view and
then L2-normalized. Each view's first-template
visual feature is also normalized. With a shared learnable scalar `alpha`
and a shared linear projection `P`, the prompt is
`P(text_view + alpha * visual_view)`. This is not a convex mixture. The visual
and text encoders are frozen; the projection and `alpha` are learnable.
Text supplies a viewpoint prior, not a target-selection command.

**PCVA.** Broadcast learnable global view tokens are added to the common
feature map. In each direction, the querying view's VLVP prompt is prepended
to the other view's key/value sequence. The adapter contains pre-normalized
attention, a feed-forward network and their internal residual connections.
Its output is scaled by a learned `gamma` and added to the base feature.
The returned feature maps are the inputs to the original model's two
prediction heads; those heads and their wiring are outside this package.

**CMD.** Gaussian target-region maps pool each view's final feature. The maximum
classification response, already sigmoid-transformed, determines a teacher
only when the inter-view confidence gap exceeds `margin`. Positive-pair cosine
embedding loss transfers the detached teacher feature to the student. Active
samples are averaged within each direction; the two directional means are
added. Target-region maps belong only to this learning objective, never to
search-frame inference. CMD is not a dense spatial L2 matching loss.

## Prompt Lifetime

The uncached `ViewpointPromptEncoder.forward(initial_template)` preserves
gradients to the fusion and projection parameters. It is the differentiable
path used for learning; calling it does not automatically read the cache.

In evaluation mode, `initialize_sequence(initial_template)` computes and
caches prompts once for **every new sequence**, after parameters are loaded.
Subsequent calls to `cached_prompts()` reuse the projected initial prompts
and make no CLIP image-encoder calls. The full tracker integration is omitted.
Updating online tracking templates does not change the initial VLVP template.
With fixed parameters, text, numerical execution settings and first template,
this is equivalent to repeatedly computing that same initial prompt.

The cache is nonpersistent and is cleared by parameter loading, device/dtype
conversion or entering training mode. Normal tensor edits to text buffers or
model parameters invalidate it; reinitialize before further inference.
Do not mutate tensors through `.data` or modify returned cached tensors.
Use one prompt-module instance per active sequence. Caching does not unload CLIP or
imply a reduction in resident model memory.

## Component Contracts

- Images are preprocessed `(B, 6, H, W)` tensors, UAV RGB first, ground RGB
  second. VLVP resizes the existing template to CLIP's input size without
  applying a second normalization.
- Template boxes use normalized `xywh`, `(B, 4)` per template and view.
- The supplied backbone provides a six-channel `patch_embed`, `blocks`,
  `num_main_blocks`, `patch_size`, `embed_dim`, `convmlp`, `stop_grad_conv1`,
  position/type embeddings, `pos_drop`, `rel_pos_bias`, `norm`, and `fc_norm`
  following the adapted Fast-iTPN interface.
- The supplied CLIP exposes `encode_image`, `encode_text`, and
  `text_projection`. The caller places all modules on compatible devices.
- Supply a nonempty list or tuple of prompt strings for each view. Empty lists
  and blank strings are rejected instead of being silently encoded.

Fast-iTPN, CLIP, and the baseline center-head components remain credited to
their original projects and subject to their respective licenses. They and
their weights are not bundled in this partial release.
