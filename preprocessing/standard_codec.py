"""Real standard video codecs and a frozen differentiable proxy.

The standard codec owns the forward values used by the loss and analyzer.  Its
FFmpeg round trip is non-differentiable, so ``ParallelStandardVideoCodec`` uses
the proxy only for the backward Jacobian:

    y = y_proxy + stop_gradient(y_real - y_proxy)

Consequently ``y == y_real`` in the forward pass while gradients with respect
to the preprocessed clip follow ``y_proxy``.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _run_ffmpeg(command: list[str]) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install it or pass --ffmpeg /path/to/ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError(f"FFmpeg failed: {detail}") from exc


def _run_ffmpeg_pipe(command: list[str], payload: bytes) -> bytes:
    try:
        completed = subprocess.run(
            command,
            input=payload,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "FFmpeg was not found. Install it or pass --ffmpeg /path/to/ffmpeg."
        ) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or b"").decode(
            "utf-8", errors="replace"
        ).strip()
        raise RuntimeError(f"FFmpeg pipe failed: {detail}") from exc
    return completed.stdout


class StandardVideoCodec(nn.Module):
    """Non-differentiable H.264/H.265 encode/decode through FFmpeg.

    Each batch item is encoded as one elementary video stream.  BPP therefore
    includes codec headers but excludes MP4/MKV container overhead.
    """

    def __init__(
        self,
        codec: str = "h264",
        qp: int = 35,
        *,
        fps: float = 30.0,
        preset: str = "medium",
        ffmpeg: str = "ffmpeg",
        io_backend: str = "pipe",
        codec_workers: int = 2,
        ffmpeg_threads: int = 1,
    ) -> None:
        super().__init__()
        if codec not in {"h264", "h265"}:
            raise ValueError("codec must be 'h264' or 'h265'")
        if not 0 <= qp <= 51:
            raise ValueError("QP must be in [0, 51]")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if io_backend not in {"pipe", "png"}:
            raise ValueError("io_backend must be 'pipe' or 'png'")
        if codec_workers < 1:
            raise ValueError("codec_workers must be >= 1")
        if ffmpeg_threads < 1:
            raise ValueError("ffmpeg_threads must be >= 1")
        self.codec = codec
        self.qp = qp
        self.fps = fps
        self.preset = preset
        self.ffmpeg = ffmpeg
        self.io_backend = io_backend
        self.codec_workers = int(codec_workers)
        self.ffmpeg_threads = int(ffmpeg_threads)

    @property
    def encoder(self) -> str:
        return "libx264" if self.codec == "h264" else "libx265"

    @property
    def extension(self) -> str:
        return ".h264" if self.codec == "h264" else ".h265"

    @property
    def format_name(self) -> str:
        return "h264" if self.codec == "h264" else "hevc"

    def set_qp(self, qp: int) -> None:
        if not 0 <= qp <= 51:
            raise ValueError("QP must be in [0, 51]")
        self.qp = qp

    @staticmethod
    def _write_frames(directory: Path, clip: torch.Tensor) -> None:
        for index, frame in enumerate(clip):
            rgb = (
                frame.detach()
                .float()
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .mul(255.0)
                .round()
                .to(torch.uint8)
                .numpy()
            )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(directory / f"input_{index:05d}.png"), bgr):
                raise RuntimeError("could not write a temporary codec input frame")

    @staticmethod
    def _read_frames(directory: Path, count: int) -> torch.Tensor:
        frames: list[torch.Tensor] = []
        for index in range(count):
            image = cv2.imread(str(directory / f"decoded_{index:05d}.png"), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"codec produced only {len(frames)} of {count} frames")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255.0))
        return torch.stack(frames)

    def _roundtrip_png(self, clip: torch.Tensor) -> tuple[torch.Tensor, float]:
        time, channels, height, width = clip.shape
        if channels != 3 or time < 1:
            raise ValueError(f"expected non-empty [T,3,H,W], got {tuple(clip.shape)}")
        if height % 2 or width % 2:
            raise ValueError(
                f"yuv420p requires even height and width, got {height}x{width}"
            )

        with tempfile.TemporaryDirectory(prefix="standard_codec_") as temporary:
            directory = Path(temporary)
            self._write_frames(directory, clip)
            stream = directory / f"stream{self.extension}"
            input_pattern = directory / "input_%05d.png"
            decoded_pattern = directory / "decoded_%05d.png"
            keyint = max(time, 1)
            command = [
                self.ffmpeg,
                "-y",
                "-loglevel",
                "error",
                "-framerate",
                str(self.fps),
                "-start_number",
                "0",
                "-i",
                str(input_pattern),
                "-frames:v",
                str(time),
                "-an",
                "-c:v",
                self.encoder,
                "-preset",
                self.preset,
                "-qp",
                str(self.qp),
                "-pix_fmt",
                "yuv420p",
                "-threads",
                str(self.ffmpeg_threads),
            ]
            if self.codec == "h264":
                command.extend(
                    ["-x264-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0"]
                )
            else:
                command.extend(
                    [
                        "-x265-params",
                        f"log-level=error:keyint={keyint}:min-keyint={keyint}:scenecut=0",
                    ]
                )
            command.extend(["-f", self.format_name, str(stream)])
            _run_ffmpeg(command)
            _run_ffmpeg(
                [
                    self.ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-threads",
                    str(self.ffmpeg_threads),
                    "-i",
                    str(stream),
                    "-frames:v",
                    str(time),
                    "-start_number",
                    "0",
                    str(decoded_pattern),
                ]
            )
            reconstructed = self._read_frames(directory, time)
            bpp = stream.stat().st_size * 8.0 / float(time * height * width)
            return reconstructed, bpp

    def _roundtrip_pipe(self, clip: torch.Tensor) -> tuple[torch.Tensor, float]:
        time, channels, height, width = clip.shape
        if channels != 3 or time < 1:
            raise ValueError(f"expected non-empty [T,3,H,W], got {tuple(clip.shape)}")
        if height % 2 or width % 2:
            raise ValueError(
                f"yuv420p requires even height and width, got {height}x{width}"
            )

        frames = (
            clip.detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        video_size = f"{width}x{height}"
        keyint = max(time, 1)
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            video_size,
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-frames:v",
            str(time),
            "-an",
            "-c:v",
            self.encoder,
            "-preset",
            self.preset,
            "-qp",
            str(self.qp),
            "-pix_fmt",
            "yuv420p",
            "-threads",
            str(self.ffmpeg_threads),
        ]
        if self.codec == "h264":
            command.extend(
                ["-x264-params", f"keyint={keyint}:min-keyint={keyint}:scenecut=0"]
            )
        else:
            command.extend(
                [
                    "-x265-params",
                    f"log-level=error:keyint={keyint}:min-keyint={keyint}:scenecut=0",
                ]
            )
        command.extend(["-f", self.format_name, "pipe:1"])
        bitstream = _run_ffmpeg_pipe(command, frames.tobytes())

        decoded = _run_ffmpeg_pipe(
            [
                self.ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                str(self.ffmpeg_threads),
                "-f",
                self.format_name,
                "-i",
                "pipe:0",
                "-frames:v",
                str(time),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            bitstream,
        )
        expected_bytes = time * height * width * channels
        if len(decoded) != expected_bytes:
            raise RuntimeError(
                "FFmpeg returned an unexpected rawvideo size: "
                f"expected {expected_bytes} bytes, received {len(decoded)}"
            )
        array = np.frombuffer(decoded, dtype=np.uint8).reshape(
            time, height, width, channels
        )
        reconstructed = (
            torch.from_numpy(array.copy()).permute(0, 3, 1, 2).float().div_(255.0)
        )
        bpp = len(bitstream) * 8.0 / float(time * height * width)
        return reconstructed, bpp

    def _roundtrip_one(self, clip: torch.Tensor) -> tuple[torch.Tensor, float]:
        if self.io_backend == "pipe":
            return self._roundtrip_pipe(clip)
        return self._roundtrip_png(clip)

    def forward(self, clip: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        device = clip.device
        dtype = clip.dtype
        # FFmpeg is outside autograd by construction.
        samples = tuple(clip.detach().cpu().unbind(0))
        if self.codec_workers > 1 and len(samples) > 1:
            worker_count = min(self.codec_workers, len(samples))
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                results = list(executor.map(self._roundtrip_one, samples))
        else:
            results = [self._roundtrip_one(sample) for sample in samples]
        reconstructions = [result[0] for result in results]
        rates = [result[1] for result in results]
        return (
            torch.stack(reconstructions).to(device=device, dtype=dtype),
            torch.tensor(rates, device=device, dtype=torch.float32),
        )


def _normalization_groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _Residual3DBlock(nn.Module):
    """Pre-activation residual block that preserves space and time resolution."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = _normalization_groups(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = self.conv1(F.gelu(self.norm1(value)))
        value = self.conv2(F.gelu(self.norm2(value)))
        return residual + value


class _FiLM3D(nn.Module):
    """QP-conditioned feature-wise affine modulation: gamma * x + beta."""

    def __init__(self, condition_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.feature_channels = feature_channels
        self.affine = nn.Linear(condition_channels, feature_channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma_delta, beta = self.affine(condition).chunk(2, dim=1)
        gamma = 1.0 + gamma_delta
        return gamma[:, :, None, None, None] * value + beta[:, :, None, None, None]


class _Conditional3DStage(nn.Module):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
        blocks: int,
    ) -> None:
        super().__init__()
        self.film = _FiLM3D(condition_channels, channels)
        self.blocks = nn.Sequential(
            *[_Residual3DBlock(channels) for _ in range(blocks)]
        )

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.film(value, condition))


class StandardCodecProxy(nn.Module):
    """FiLM-conditioned deep 3-D skip proxy distilled from a standard codec."""

    ARCHITECTURE = "film_deeper3d_v1"

    def __init__(
        self,
        hidden_channels: int = 48,
        latent_channels: int = 64,
        bottleneck_channels: int = 96,
        blocks_per_stage: int = 2,
        film_channels: int = 64,
        qp_step_divisor: float = 12.0,
        max_delta: float = 0.5,
    ) -> None:
        super().__init__()
        if min(hidden_channels, latent_channels, bottleneck_channels, film_channels) < 1:
            raise ValueError("proxy channel counts must be positive")
        if blocks_per_stage < 1:
            raise ValueError("blocks_per_stage must be positive")
        if qp_step_divisor <= 0:
            raise ValueError("qp_step_divisor must be positive")
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_channels
        self.bottleneck_channels = bottleneck_channels
        self.blocks_per_stage = blocks_per_stage
        self.film_channels = film_channels
        self.qp_step_divisor = qp_step_divisor
        self.max_delta = max_delta

        self.qp_embedding = nn.Sequential(
            nn.Linear(1, film_channels),
            nn.GELU(),
            nn.Linear(film_channels, film_channels),
            nn.GELU(),
        )
        self.stem = nn.Conv3d(
            3,
            hidden_channels,
            kernel_size=(3, 5, 5),
            stride=(1, 2, 2),
            padding=(1, 2, 2),
        )
        self.encoder_high = _Conditional3DStage(
            hidden_channels, film_channels, blocks_per_stage
        )
        self.down_middle = nn.Conv3d(
            hidden_channels,
            latent_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.encoder_middle = _Conditional3DStage(
            latent_channels, film_channels, blocks_per_stage
        )
        self.down_bottleneck = nn.Conv3d(
            latent_channels,
            bottleneck_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.bottleneck = _Conditional3DStage(
            bottleneck_channels, film_channels, blocks_per_stage
        )

        self.up_middle = nn.ConvTranspose3d(
            bottleneck_channels,
            latent_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.decoder_middle = _Conditional3DStage(
            latent_channels, film_channels, blocks_per_stage
        )
        self.up_high = nn.ConvTranspose3d(
            latent_channels,
            hidden_channels,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        self.decoder_high = _Conditional3DStage(
            hidden_channels, film_channels, blocks_per_stage
        )
        self.to_rgb = nn.ConvTranspose3d(
            hidden_channels,
            3,
            kernel_size=(3, 4, 4),
            stride=(1, 2, 2),
            padding=(1, 1, 1),
        )
        nn.init.zeros_(self.to_rgb.weight)
        nn.init.zeros_(self.to_rgb.bias)

        rate_features = bottleneck_channels * 4 + 1
        self.rate_head = nn.Sequential(
            nn.Linear(rate_features, bottleneck_channels * 2),
            nn.GELU(),
            nn.Linear(bottleneck_channels * 2, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, 1),
        )

    @property
    def config(self) -> dict[str, Any]:
        return {
            "architecture": self.ARCHITECTURE,
            "hidden_channels": self.hidden_channels,
            "latent_channels": self.latent_channels,
            "bottleneck_channels": self.bottleneck_channels,
            "blocks_per_stage": self.blocks_per_stage,
            "film_channels": self.film_channels,
            "qp_step_divisor": self.qp_step_divisor,
            "max_delta": self.max_delta,
        }

    @staticmethod
    def _qp_tensor(
        qp: int | float | torch.Tensor,
        batch: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        value = torch.as_tensor(qp, device=device, dtype=dtype).flatten()
        if value.numel() == 1:
            value = value.expand(batch)
        if value.numel() != batch:
            raise ValueError(f"QP must be scalar or have {batch} elements")
        return value

    @staticmethod
    def _ste_round(value: torch.Tensor) -> torch.Tensor:
        return value + (value.round() - value).detach()

    def _rate_statistics(self, quantized: torch.Tensor) -> torch.Tensor:
        mean_magnitude = quantized.abs().mean(dim=(2, 3, 4))
        spatial_variance = quantized.var(dim=(3, 4), unbiased=False).mean(dim=2)
        smooth_sparsity = torch.exp(-quantized.abs()).mean(dim=(2, 3, 4))
        if quantized.shape[2] > 1:
            temporal_change = (
                quantized[:, :, 1:] - quantized[:, :, :-1]
            ).abs().mean(dim=(2, 3, 4))
        else:
            temporal_change = torch.zeros_like(mean_magnitude)
        return torch.cat(
            (mean_magnitude, spatial_variance, smooth_sparsity, temporal_change),
            dim=1,
        )

    def forward(
        self, clip: torch.Tensor, qp: int | float | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if clip.ndim != 5 or clip.shape[2] != 3:
            raise ValueError(f"expected [B,T,3,H,W], got {tuple(clip.shape)}")
        batch, time, _, height, width = clip.shape
        pad_height = (8 - height % 8) % 8
        pad_width = (8 - width % 8) % 8
        channels_first = clip.permute(0, 2, 1, 3, 4)
        if pad_height or pad_width:
            channels_first = F.pad(
                channels_first,
                (0, pad_width, 0, pad_height, 0, 0),
                mode="replicate",
            )

        qp_value = self._qp_tensor(
            qp, batch, device=clip.device, dtype=clip.dtype
        )
        qp_normalized = ((qp_value - 25.5) / 25.5).unsqueeze(1)
        condition = self.qp_embedding(qp_normalized)

        high = self.encoder_high(self.stem(channels_first), condition)
        middle = self.encoder_middle(self.down_middle(high), condition)
        bottleneck = self.bottleneck(self.down_bottleneck(middle), condition)
        q_step = torch.pow(
            2.0, (qp_value - 32.0) / self.qp_step_divisor
        ).view(batch, 1, 1, 1, 1)
        quantized = self._ste_round(bottleneck / q_step) * q_step

        decoded_middle = self.up_middle(quantized) + middle
        decoded_middle = self.decoder_middle(decoded_middle, condition)
        decoded_high = self.up_high(decoded_middle) + high
        decoded_high = self.decoder_high(decoded_high, condition)
        delta = self.max_delta * torch.tanh(self.to_rgb(decoded_high))
        delta = delta[..., :time, :height, :width]
        source = channels_first[..., :time, :height, :width]
        reconstruction = (source + delta).clamp(0.0, 1.0)
        reconstruction = reconstruction.permute(0, 2, 1, 3, 4)

        statistics = self._rate_statistics(quantized)
        rate_features = torch.cat((statistics, qp_normalized), dim=1)
        rate = F.softplus(self.rate_head(rate_features)).squeeze(1)
        return reconstruction, rate

    @classmethod
    def from_checkpoint(
        cls, path: str | Path, map_location: str | torch.device = "cpu"
    ) -> StandardCodecProxy:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
        if not isinstance(checkpoint, dict):
            raise ValueError(f"unsupported proxy checkpoint: {path}")
        config = dict(checkpoint.get("proxy_config", {}))
        architecture = config.pop("architecture", None)
        if architecture != cls.ARCHITECTURE:
            raise ValueError(
                f"proxy checkpoint architecture is {architecture or 'legacy_shallow'}, "
                f"expected {cls.ARCHITECTURE}; retrain the proxy from scratch"
            )
        model = cls(**config)
        state = checkpoint.get("proxy", checkpoint.get("state_dict", checkpoint))
        model.load_state_dict(state)
        return model


class ParallelStandardVideoCodec(nn.Module):
    """Real codec forward pass with a frozen proxy providing backward gradients."""

    def __init__(self, standard_codec: nn.Module, proxy: StandardCodecProxy) -> None:
        super().__init__()
        self.standard_codec = standard_codec
        self.proxy = proxy.requires_grad_(False)
        self.proxy.eval()

    @property
    def qp(self) -> int:
        return int(getattr(self.standard_codec, "qp"))

    def set_qp(self, qp: int) -> None:
        setter = getattr(self.standard_codec, "set_qp", None)
        if setter is None:
            raise TypeError("standard codec does not implement set_qp")
        setter(qp)

    def train(self, mode: bool = True) -> ParallelStandardVideoCodec:
        super().train(mode)
        self.proxy.eval()
        return self

    def forward(
        self, clip: torch.Tensor, *, use_proxy_gradient: bool | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_reconstruction, real_bpp = self.standard_codec(clip.detach())
        if use_proxy_gradient is None:
            use_proxy_gradient = self.training and torch.is_grad_enabled()
        if not use_proxy_gradient:
            return real_reconstruction, real_bpp

        proxy_reconstruction, proxy_bpp = self.proxy(clip, self.qp)
        reconstruction = proxy_reconstruction + (
            real_reconstruction - proxy_reconstruction
        ).detach()
        bpp = proxy_bpp + (real_bpp - proxy_bpp).detach()
        return reconstruction, bpp


def require_ffmpeg(executable: str = "ffmpeg") -> None:
    """Fail before a long run if the configured FFmpeg executable is missing."""

    if Path(executable).parent != Path("."):
        available = Path(executable).is_file()
    else:
        available = shutil.which(executable) is not None
    if not available:
        raise RuntimeError(f"FFmpeg executable not found: {executable!r}")
