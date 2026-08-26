# Video Swin Lite v2 with clean distillation and Task BD-rate selection

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

## V2 objective

```text
L = alpha * (L_D_hybrid + lambda * L_R)
    + w_CE * CE(labels)
    + w_KD * KD(reconstruction, clean video)
    + w_F * cosine_feature_loss(layer4)
```

- `L_D_hybrid = eta*MSE(reconstruction, source) + (1-eta)*MSE(processed, source)`.
- `L_R`: measured elementary-stream BPP from H.264/H.265.
- `KD`: temperature-scaled KL divergence from frozen R3D-18 clean-video logits.
- Feature matching uses channel-normalized `layer4` activations from the same
  frozen analyzer. Clean targets are detached; gradients pass through the
  reconstruction/analyzer/proxy only to Video Swin Lite.

V2 defaults are `eta=0.25`, `w_CE=1`, `w_KD=0.5`, `T=2`, and `w_F=0.05`.
Set `--kd-weight 0 --feature-weight 0 --distortion-reconstruction-weight 1`
to reproduce the v1 loss. Optional `--normalize-rate-by-anchor` divides BPP by
the fixed validation anchor at the current QP; this is a relative-rate surrogate,
not the BD-rate integral, and needs a smaller separately tuned lambda.

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

Then train Video Swin Lite through the real codec and frozen proxy:

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
  --codec h264 \
  --codec-qps 30 35 40 45 \
  --qp-sampling-weights 0.15 0.25 0.30 0.30 \
  --distortion-reconstruction-weight 0.25 \
  --ce-weight 1.0 \
  --kd-weight 0.5 \
  --kd-temperature 2.0 \
  --feature-weight 0.05 \
  --feature-layer layer4 \
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
- `preprocessing/evaluation.py`: reproducible held-out split and BD-rate helpers.
- `model_summary.py`: torchinfo summaries for preprocessor and proxy.
- `evaluate_real_codec.py`: real-codec metrics, Top-1/BPP plots and BD-rate.
- `visualize_pipeline.py`: qualitative output from the same held-out split.
