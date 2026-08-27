# Video Swin Lite v3: masked-rate preprocessing with Task BD-rate selection

Task-aware video preprocessing with the requested pipeline kept intact:

```text
video -> Video Swin Lite -> H.264/H.265 -> reconstruction
      -> frozen analyzer -> task
```

During training, a frozen differentiable proxy runs beside the real codec:

```text
                            +-> frozen codec proxy -- backward gradients --+
                            |                                               |
video -> trainable preprocessor -> real H.264/H.265 -> reconstruction ------+
                                                    -> frozen analyzer -> task
```

The real FFmpeg codec always determines reconstruction and measured BPP in the
forward pass. The proxy supplies only the backward Jacobian:

```python
reconstruction = proxy_reconstruction + (
    real_reconstruction - proxy_reconstruction
).detach()

bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
```

Consequently, task and rate-distortion loss values correspond to the standard
codec while gradients still reach only the preprocessor. Codec proxy and analyzer
parameters remain frozen.

## FiLM deeper-3D proxy

The differentiable proxy is a compact 3-D residual encoder-decoder rather than
the earlier two-layer Conv3D model:

```text
RGB clip + per-sample QP
  -> spatial /2 Conv3D -> FiLM -> residual 3-D blocks ------------ skip ----+
  -> spatial /4 Conv3D -> FiLM -> residual 3-D blocks ------ skip ----+      |
  -> spatial /8 Conv3D -> FiLM -> residual 3-D bottleneck           |      |
  -> softened STE quantization                                      |      |
  -> up /4 + skip -> FiLM -> residual 3-D blocks <------------------+      |
  -> up /2 + skip -> FiLM -> residual 3-D blocks <-------------------------+
  -> RGB residual -> input + delta -> proxy reconstruction
```

Every FiLM layer applies `gamma(QP) * feature + beta(QP)`, so mixed-QP samples
can change feature scale as well as bias. The three spatial scales and residual
blocks give a receptive field larger than a 64x64 H.264 coding-tree region.
The rate head uses mean magnitude, spatial variance, smooth sparsity and temporal
change statistics. The proxy remains frozen during preprocessor training, but
autograd still differentiates its output with respect to the preprocessed clip.

## Video Swin Lite preprocessor

`VideoSwinLitePreprocessor` is a compact dense video transformer:

```text
BTCHW RGB video + codec QP
  -> normalized QP embedding (MLP)
  -> Conv3D spatial patch embedding, patch=(1,4,4), 3 -> 48 channels
  -> depthwise Conv3D positional encoding
  -> four alternating regular/shifted 3-D Swin blocks
       QP FiLM: (1 + gamma(QP)) * feature + beta(QP)
       window=(4,8,8), heads=4, MLP ratio=4
  -> LayerNorm
  -> ConvTranspose3D spatial reconstruction, 48 -> RGB
  -> tanh * 0.25 * sigmoid(QP residual gate)
  -> input + RGB residual
```

Temporal resolution is never downsampled. Shifted windows exchange information
between neighboring clips and spatial regions while avoiding global space-time
attention. The RGB head is zero-initialized, so a new model is exactly the identity
mapping. The earlier factorized ViT and CNN remain available as `--preprocessor vit`
and `--preprocessor cnn` for ablation; `swin` is the default.
QP conditioning is enabled by default for new Swin checkpoints. It lets low QPs
suppress expensive texture without forcing high QPs to use the same residual.
Evaluation reconstructs the QP-specific preprocessed clip before every real-codec
operating point. Legacy checkpoints without QP parameters remain loadable.

## V3 objective

```text
L = alpha * (L_D_hybrid + lambda * L_R)
    + w_CE * CE(labels)
    + w_KD * KD(reconstruction, clean video)
    + w_F * cosine_feature_loss(layer4)
    + w_M * masked_total_variation(preprocessed clip)
```

- `L_D_hybrid = eta*MSE(reconstruction, source) + (1-eta)*MSE(processed, source)`.
- `L_R`: measured elementary-stream BPP from H.264/H.265.
- `KD`: temperature-scaled KL divergence from frozen R3D-18 clean-video logits.
- Feature matching uses channel-normalized `layer4` activations from the same
  frozen analyzer. Clean targets are detached; gradients pass through the
  reconstruction/analyzer/proxy only to Video Swin Lite.
- The masked term is new in v3 and is described below. It defaults to zero, so an
  unchanged command line reproduces the v2 objective exactly.
- Optional `--normalize-rate-by-anchor` divides BPP by the fixed validation anchor
  at the current QP. It is a relative-rate surrogate, not the BD-rate integral, and
  needs a smaller separately tuned lambda.

### The masked rate penalty

Every measured run so far *increased* bitrate: the preprocessor learned to be an
enhancement filter, buying Top-1 with bits. `lambda * bpp` cannot fix that. Over
QP 30-45 the whole rate term spans roughly 0.010-0.012, so raising lambda scales a
term with almost no dynamic range, and it reaches the preprocessor only through the
frozen proxy's rate head.

`--mask-rate-weight` adds a term that has both properties the scalar lacks. It is
spatially resolved, and it is an exact analytic gradient with no proxy in the path:

```text
w_M * weighted_mean( m_ij * ( |dx z| + |dy z| + |dt z| ) )
m = 1 - normalized(|clean layer4 activation|)   inside the analyzer crop
m = --mask-rate-outside-weight                  outside it
```

H.264 spends bits on spatial and temporal detail, so an L1 penalty on the gradients
of the preprocessed clip is a differentiable stand-in for bitrate that a per-pixel
weight can steer. The weight comes from the frozen analyzer's own activation energy
on the *clean* clip, so it is a detached constant: regions the analyzer relies on
are left alone, and quiet regions become cheap to flatten. Each finite difference
takes the smaller of its two endpoint weights, so detail bordering a protected
pixel stays protected.

Two options control how aggressive it is:

- `--mask-rate-target output` (default) penalizes detail in the preprocessed clip,
  so it can push bitrate below the anchor. `residual` penalizes only `z - x`, which
  stops the preprocessor from *adding* detail but can never remove any.
- `--mask-rate-outside-weight` covers the border the analyzer never sees. On
  128x128 input the `r3d_18` preset resizes to 128x171 and center-crops 112x112, so
  the analyzer only sees rows 8:120 and columns 22:106, that is **57.4% of every
  frame**. The other 42.6% costs bits that cannot change the prediction. Training
  prints this box at startup. Flattening it is a real saving under this evaluation
  protocol, and it is also the part a reader is most likely to call exploiting the
  evaluation crop, so report it explicitly or set the flag to `0.0`.

Choose `w_M` by comparing gradients, not loss values. At MSE 0.002 the mean absolute
reconstruction error is about 0.045, so `alpha * d(MSE)/dz` is near
`2 * 10 * 0.045 = 0.9` per pixel; `--mask-rate-weight 1.0` applies comparable
pressure. Sweep 0.25, 1, 4. The logged `mask_rate` value itself sits near 0.05-0.15
for real video, between a smooth gradient (0.07) and uniform noise (0.33).

Set `--distortion-reconstruction-weight 1.0` when the masked term is on. `eta < 1`
adds `MSE(processed, source)`, whose minimum is at `processed == source`; it is an
identity regularizer, not a smoothing incentive, and it pulls directly against the
masked penalty.

### Presets and the experiment ladder

`train.py` accepts `@file` presets, one flag per line, with `#` comments allowed:

```bash
python -u train.py @presets/v1_parity.args \
  --data-root /path/to/kinetics/train \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --output-dir checkpoints/v1_parity
```

| Preset | Objective |
| --- | --- |
| `presets/v1_parity.args` | control run: `eta=1`, no KD, no feature loss, no masked term |
| `presets/v2_distill.args` | v2 defaults: `eta=0.25`, KD 0.5, feature 0.05 |
| `presets/v3_masked_rate.args` | `eta=1`, KD 0.5, feature 0.05, masked TV on the output |
| `presets/v3_masked_rate_conservative.args` | masked TV on the residual only, inside the crop only |

Run the control first. The v2 defaults already changed the loss, so a v2 or v3
number cannot be compared against an older run until the control has been measured
in this repository with the same data limits and epoch count.

### QP sampling

Leave `--qp-sampling-weights` unset. Uniform sampling is the default and the
recommendation, for three separate reasons:

1. With four QPs, `calculate_bd_rate` fits `degree = min(3, len(quality)-1) = 3`
   through four points. That is exact interpolation with zero degrees of freedom,
   and the per-QP leverage ordering flips between a degree-2 and a degree-3 fit. A
   sampling tilt therefore optimizes a property of the fit, not of the curve.
2. Cross-entropy is already largest at QP 45, so the gradient is self-weighted
   toward high QP. Adding a sampling tilt double-counts that.
3. The BD-rate integral runs over the shared quality window
   `[Top1_proposed(QP45), Top1_anchor(QP30)]`. Gains at QP 45 raise the lower bound
   of that window, and most of its content sits in the QP 30-40 band, so starving
   QP 30 removes training signal from the region actually being measured.

Training prints the effective sampling distribution at startup.

### Reading the numbers

`--limit-val 400` is a pilot setting, not a measurement. On this dataset a 400-clip
validation subset reproduces `bpp` to about 0.06 points but inflates Top-1 by about
0.69 points, which is roughly 2.3 BD points. Use the limited run to rank epochs and
configurations, then re-measure the winner on the full split before reporting it.
Training prints a warning when `--limit-val` and `--checkpoint-metric task_bd_rate`
are combined.

The measured exchange rate on this pipeline is 1 Top-1 point per 3.3 BD points and
1 bpp point per 0.95 BD points, so accuracy is worth about 3.4 times bitrate. A run
that trades 1 point of Top-1 for 3% of bitrate is a net loss.

## Requirements

```bash
pip install -r requirements.txt
```

FFmpeg must include `libx264` and/or `libx265`. There is one unified requirements
file; no Kaggle-specific requirements file is needed.

## Training order

First build a deterministic codec cache. The raw pipe is checked against the
legacy PNG path before caching, two FFmpeg workers run concurrently, and each
source clip is stored only once as `uint8`. `train/` and `val/` have separate
cache trees; when the dataset has no `val/`, the split is stratified and fixed by
`--seed`.

```bash
python -u precompute_codec.py \
  --data-root /path/to/kinetics/train \
  --codec h264 \
  --qps 30 35 40 45 \
  --codec-io pipe \
  --codec-workers 2 \
  --output-dir precomputed_codec/h264
```

Then distill a codec-specific proxy without invoking FFmpeg in every epoch:

```bash
python -u train_proxy.py \
  --precomputed-root precomputed_codec/h264 \
  --codec h264 \
  --qps 30 35 40 45 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 20 \
  --batch-size 8 \
  --hidden-channels 48 \
  --latent-channels 64 \
  --bottleneck-channels 96 \
  --blocks-per-stage 2 \
  --film-channels 64 \
  --qp-step-divisor 12 \
  --clip-grad 1.0 \
  --scheduler-factor 0.5 \
  --scheduler-patience 3 \
  --output-dir checkpoints/h264_proxy
```

Every cached training batch is balanced across the four QPs. Batch sizes 8 or
16 are recommended. Validation always evaluates the fixed cached split. The
legacy online path remains available by replacing `--precomputed-root` with
`--data-root`; it now uses raw pipes and two codec workers by default.

The proxy architecture changed, so a shallow-proxy `last.pt` cannot be resumed.
Start FiLM deeper-3D training at epoch 1 with a new output directory. Existing
precomputed codec caches remain fully reusable because their real reconstruction
and BPP targets are architecture-independent.

Then train Video Swin Lite through the real codec and frozen proxy. A preset carries
the objective; the command line carries only the environment:

```bash
python -u train.py @presets/v3_masked_rate.args \
  --data-root /path/to/kinetics/train \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --codec-fps 30 \
  --codec-preset medium \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --workers 4 \
  --output-dir checkpoints/preprocessor
```

The same run written out in full, without a preset:

```bash
python -u train.py \
  --data-root /path/to/kinetics/train \
  --proxy-checkpoint checkpoints/h264_proxy/best.pt \
  --preprocessor swin \
  --swin-patch-size 4 \
  --swin-embed-dim 48 \
  --swin-depth 4 \
  --swin-heads 4 \
  --swin-window-temporal 4 \
  --swin-window-spatial 8 \
  --swin-qp-conditioning \
  --swin-qp-embed-dim 64 \
  --max-residual 0.25 \
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --distortion-reconstruction-weight 1.0 \
  --ce-weight 1.0 \
  --kd-weight 0.5 \
  --kd-temperature 2.0 \
  --feature-weight 0.05 \
  --feature-layer layer4 \
  --mask-rate-weight 1.0 \
  --mask-rate-target output \
  --mask-rate-layer layer4 \
  --optimizer adamw \
  --weight-decay 0.01 \
  --alpha 10 \
  --rate-lambda 0.05 \
  --frames 16 \
  --frame-stride 2 \
  --frame-size 128 \
  --epochs 30 \
  --batch-size 1 \
  --accumulation-steps 4 \
  --checkpoint-metric task_bd_rate \
  --output-dir checkpoints/preprocessor
```

If `val/` is absent, training creates a deterministic stratified validation subset
using indices in memory. `--limit-train` and `--limit-val` now use a deterministic
approximately class-balanced subset instead of slicing the globally shuffled list.
Classes containing only one video remain in training.

The anchor H.264 curve is measured once and cached as `anchor_validation.json`.
Every epoch evaluates every validation clip at all four QPs and reports Top-1
BD-rate. Checkpoints are written independently as `best_loss.pt`, `best_ce.pt`,
`best_top1.pt`, and `best_task_bd_rate.pt`; `best.pt` follows
`--checkpoint-metric` and defaults to Task BD-rate. Resume restores optimizer,
scheduler and AMP scaler state.

## Real-codec evaluation

Final evaluation never uses the proxy. It compares the anchor and preprocessed
clips through the real FFmpeg codec and frozen analyzer. If `val/` is absent,
`evaluate_real_codec.py` automatically recreates the checkpoint's stratified
validation split in memory from its saved `val_ratio` and `seed`; no validation
folder, symlinks or precomputed codec cache are required.

```bash
python -u evaluate_real_codec.py \
  --checkpoint checkpoints/preprocessor/best.pt \
  --data-root /path/to/kinetics/train \
  --codecs h264 \
  --qps 30 35 40 45 \
  --device cuda \
  --output-dir outputs/real_codec
```

The output includes `metrics.csv`, `metrics.json`, `clean_metrics.json`,
`bd_rate.json`, a focused `<codec>_top1_bd_rate.png`, and the three-panel
`<codec>_top1_bpp_bd_rate.png`. Task BD-rate uses Top-1 as the
quality axis; PSNR BD-rate is also reported. Negative BD-rate means bitrate
saving at equal quality. Task BD-rate is reported as undefined when discrete
Top-1 curves have too few distinct points or no overlapping accuracy range.

Omit `--limit` for the final result. `--limit 200` is useful for a faster pilot,
but produces noisier Top-1 and task BD-rate estimates. The limit applies only to
evaluation videos after the deterministic split and is independent of the proxy
precompute limits.

## Kaggle and model summary

Ready-to-run Kaggle cells are in [KAGGLE_GUIDE_VI.md](KAGGLE_GUIDE_VI.md).

```bash
python model_summary.py \
  --model all \
  --preprocessor swin \
  --frames 16 \
  --frame-size 128 \
  --device auto
```

## Main files

- `preprocessing/swin.py`: Video Swin Lite and 3-D shifted-window attention.
- `preprocessing/model.py`: preprocessor factory plus factorized ViT/CNN ablations.
- `preprocessing/standard_codec.py`: FFmpeg codecs, proxy and gradient bridge.
- `precompute_codec.py`: deterministic train/val uint8 codec cache and pipe verification.
- `train_proxy.py`: distill the proxy from real codec outputs and measured BPP.
- `train.py`: train only the preprocessor with rate-distortion-task loss.
- `presets/*.args`: the objective of each rung of the experiment ladder.
- `preprocessing/evaluation.py`: reproducible held-out split and BD-rate helpers.
- `model_summary.py`: torchinfo summaries for preprocessor and proxy.
- `evaluate_real_codec.py`: real-codec metrics, Top-1/BPP plots and BD-rate.
- `visualize_pipeline.py`: qualitative output from the same held-out split.
