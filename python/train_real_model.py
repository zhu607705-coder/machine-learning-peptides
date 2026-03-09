from __future__ import annotations

import contextlib
import json
import os
import platform
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .losses import DEFAULT_HUBER_DELTA, build_torch_huber_loss
    from .real_data import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        REAL_DATASET_SOURCE,
        REAL_DATASET_URL,
        TARGET_COLUMNS,
        TOKEN_ALPHABET,
        TOKEN_TO_INDEX,
        encode_token_sequence,
        load_real_dataset,
        project_root,
    )
except ImportError:
    from losses import DEFAULT_HUBER_DELTA, build_torch_huber_loss
    from real_data import (
        CATEGORICAL_COLUMNS,
        NUMERIC_COLUMNS,
        REAL_DATASET_SOURCE,
        REAL_DATASET_URL,
        TARGET_COLUMNS,
        TOKEN_ALPHABET,
        TOKEN_TO_INDEX,
        encode_token_sequence,
        load_real_dataset,
        project_root,
    )


SPLIT_SEED = 20260302
TRAINING_SEED = 20260305
HUBER_DELTA = DEFAULT_HUBER_DELTA
PREVIOUS_TARGET_COLUMNS = ["prev_area", "prev_height", "prev_width", "prev_diff"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str
    max_length: int
    embed_dim: int
    sequence_hidden: int
    numeric_hidden: int
    trunk_hidden: int
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    patience: int
    huber_delta: float = HUBER_DELTA
    include_prev_targets: bool = False
    predict_delta: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    device_type: str
    device_name: str
    selection_reason: str
    cpu_threads: int
    interop_threads: int
    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking: bool
    use_amp: bool
    amp_dtype: str | None
    compile_model: bool
    compile_mode: str | None


class StepDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_length: int,
        numeric_mean: np.ndarray | None = None,
        numeric_std: np.ndarray | None = None,
        target_mean: np.ndarray | None = None,
        target_std: np.ndarray | None = None,
        include_prev_targets: bool = False,
        predict_delta: bool = False,
        fit: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.max_length = max_length
        self.include_prev_targets = include_prev_targets
        self.predict_delta = predict_delta

        sequence_tokens = np.array(
            [encode_token_sequence(value, max_length) for value in self.df["pre-chain"].tolist()],
            dtype=np.int64,
        )
        next_tokens = np.array(
            [TOKEN_TO_INDEX.get(value, 0) for value in self.df["amino_acid"].tolist()],
            dtype=np.int64,
        )
        coupling_tokens = np.array(
            [0 if value == "HATU" else 1 for value in self.df["coupling_agent"].tolist()],
            dtype=np.int64,
        )

        numeric_columns = [*NUMERIC_COLUMNS, "sequence_length"]
        if include_prev_targets:
            numeric_columns.extend(PREVIOUS_TARGET_COLUMNS)

        numeric_features = self.df[numeric_columns].to_numpy(dtype=np.float32)
        targets = self.df[TARGET_COLUMNS].to_numpy(dtype=np.float32)
        prev_targets = self.df[PREVIOUS_TARGET_COLUMNS].to_numpy(dtype=np.float32)
        training_targets = targets - prev_targets if predict_delta else targets

        if fit:
            numeric_mean = numeric_features.mean(axis=0)
            numeric_std = numeric_features.std(axis=0)
            target_mean = training_targets.mean(axis=0)
            target_std = training_targets.std(axis=0)

        assert numeric_mean is not None
        assert numeric_std is not None
        assert target_mean is not None
        assert target_std is not None

        self.numeric_mean = numeric_mean.astype(np.float32)
        self.numeric_std = np.maximum(numeric_std, 1e-6).astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std = np.maximum(target_std, 1e-6).astype(np.float32)

        numeric = ((numeric_features - self.numeric_mean) / self.numeric_std).astype(np.float32)
        targets_normalized = ((training_targets - self.target_mean) / self.target_std).astype(np.float32)

        # Materialize tensors once to reduce per-sample overhead on both CPU and GPU runs.
        self.sequence_tokens = torch.from_numpy(sequence_tokens)
        self.next_tokens = torch.from_numpy(next_tokens)
        self.coupling_tokens = torch.from_numpy(coupling_tokens)
        self.numeric = torch.from_numpy(np.ascontiguousarray(numeric))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets_normalized))
        self.raw_targets = torch.from_numpy(np.ascontiguousarray(targets))
        self.prev_targets = torch.from_numpy(np.ascontiguousarray(prev_targets))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "sequence_tokens": self.sequence_tokens[index],
            "next_tokens": self.next_tokens[index],
            "coupling_tokens": self.coupling_tokens[index],
            "numeric": self.numeric[index],
            "targets": self.targets[index],
            "raw_targets": self.raw_targets[index],
            "prev_targets": self.prev_targets[index],
        }


class SequenceEncoder(nn.Module):
    def __init__(self, architecture: str, vocab_size: int, embed_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)

        if architecture == "gru":
            self.encoder = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.output_dim = hidden_dim * 6
        elif architecture == "cnn":
            self.conv3 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
            self.output_dim = hidden_dim * 4
        elif architecture == "hybrid":
            self.gru = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.conv3 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
            self.conv5 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2)
            self.output_dim = hidden_dim * 8
        elif architecture == "rnn_attention":
            self.rnn = nn.GRU(
                input_size=embed_dim,
                hidden_size=hidden_dim,
                batch_first=True,
                bidirectional=True,
            )
            self.attn_proj = nn.Linear(hidden_dim * 2, hidden_dim)
            self.attn_score = nn.Linear(hidden_dim, 1, bias=False)
            self.output_dim = hidden_dim * 6
        else:
            raise ValueError(f"Unsupported sequence architecture: {architecture}")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))
        mask = (tokens != 0).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)

        if self.architecture == "gru":
            output, hidden = self.encoder(embedded)
            masked_output = output * mask.unsqueeze(-1)
            mean_pool = masked_output.sum(dim=1) / lengths.unsqueeze(-1)
            max_pool = masked_output.masked_fill(mask.unsqueeze(-1) == 0, -1e9).amax(dim=1)
            hidden = hidden.transpose(0, 1).reshape(tokens.size(0), -1)
            return torch.cat([mean_pool, max_pool, hidden], dim=1)

        if self.architecture in {"cnn", "hybrid"}:
            conv_input = embedded.transpose(1, 2)
            conv3 = torch.relu(self.conv3(conv_input))
            conv5 = torch.relu(self.conv5(conv_input))
            cnn_features = torch.cat([conv3, conv5], dim=1)
            masked_features = cnn_features * mask.unsqueeze(1)
            mean_pool = masked_features.sum(dim=2) / lengths.unsqueeze(1)
            max_pool = masked_features.masked_fill(mask.unsqueeze(1) == 0, -1e9).amax(dim=2)

            if self.architecture == "cnn":
                return torch.cat([mean_pool, max_pool], dim=1)

            output, hidden = self.gru(embedded)
            masked_output = output * mask.unsqueeze(-1)
            gru_mean = masked_output.sum(dim=1) / lengths.unsqueeze(-1)
            hidden = hidden.transpose(0, 1).reshape(tokens.size(0), -1)
            return torch.cat([mean_pool, max_pool, gru_mean, hidden], dim=1)

        output, _ = self.rnn(embedded)
        masked_output = output * mask.unsqueeze(-1)
        rnn_mean = masked_output.sum(dim=1) / lengths.unsqueeze(-1)
        rnn_max = masked_output.masked_fill(mask.unsqueeze(-1) == 0, -1e9).amax(dim=1)
        attn_logits = self.attn_score(torch.tanh(self.attn_proj(output))).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_context = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)
        return torch.cat([rnn_mean, rnn_max, attn_context], dim=1)


class ConditionalAttentionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, condition_dim: int, dropout: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.query_proj = nn.Linear(condition_dim, hidden_dim * 2)
        self.state_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.attn_score = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.output_dim = hidden_dim * 8

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(tokens))
        mask = (tokens != 0).float()
        lengths = mask.sum(dim=1).clamp(min=1.0)

        output, _ = self.rnn(embedded)
        masked_output = output * mask.unsqueeze(-1)
        seq_mean = masked_output.sum(dim=1) / lengths.unsqueeze(-1)
        seq_max = masked_output.masked_fill(mask.unsqueeze(-1) == 0, -1e9).amax(dim=1)

        query = torch.tanh(self.query_proj(condition))
        attn_logits = self.attn_score(torch.tanh(self.state_proj(output) + query.unsqueeze(1))).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_context = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)
        return torch.cat([seq_mean, seq_max, attn_context, query], dim=1)


class ResidualRegressor(nn.Module):
    def __init__(self, config: ModelConfig, numeric_dim: int, output_dim: int) -> None:
        super().__init__()
        self.architecture = config.architecture
        self.next_embedding = nn.Embedding(len(TOKEN_ALPHABET), config.embed_dim, padding_idx=0)
        self.coupling_embedding = nn.Embedding(2, 4)
        self.numeric_branch = nn.Sequential(
            nn.LayerNorm(numeric_dim),
            nn.Linear(numeric_dim, config.numeric_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.numeric_hidden, config.numeric_hidden),
            nn.GELU(),
        )

        condition_dim = config.embed_dim + 4 + config.numeric_hidden
        if config.architecture == "conditional_gru_attention":
            self.sequence_encoder = ConditionalAttentionEncoder(
                vocab_size=len(TOKEN_ALPHABET),
                embed_dim=config.embed_dim,
                hidden_dim=config.sequence_hidden,
                condition_dim=condition_dim,
                dropout=config.dropout,
            )
        else:
            self.sequence_encoder = SequenceEncoder(
                architecture=config.architecture,
                vocab_size=len(TOKEN_ALPHABET),
                embed_dim=config.embed_dim,
                hidden_dim=config.sequence_hidden,
                dropout=config.dropout,
            )

        total_dim = self.sequence_encoder.output_dim + condition_dim
        self.trunk = nn.Sequential(
            nn.Linear(total_dim, config.trunk_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.trunk_hidden, config.trunk_hidden),
            nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(config.trunk_hidden, config.trunk_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.trunk_hidden, config.trunk_hidden),
        )
        if config.architecture == "conditional_gru_attention":
            head_hidden = max(8, config.trunk_hidden // 2)
            self.target_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(config.trunk_hidden, head_hidden),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(head_hidden, 1),
                    )
                    for _ in range(output_dim)
                ]
            )
            self.head = None
        else:
            self.target_heads = None
            self.head = nn.Linear(config.trunk_hidden, output_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        next_features = self.next_embedding(batch["next_tokens"])
        coupling_features = self.coupling_embedding(batch["coupling_tokens"])
        numeric_features = self.numeric_branch(batch["numeric"])
        condition_features = torch.cat([next_features, coupling_features, numeric_features], dim=1)

        if self.architecture == "conditional_gru_attention":
            sequence_features = self.sequence_encoder(batch["sequence_tokens"], condition_features)
        else:
            sequence_features = self.sequence_encoder(batch["sequence_tokens"])

        features = torch.cat([sequence_features, condition_features], dim=1)
        trunk = self.trunk(features)
        residual = torch.relu(self.residual(trunk) + trunk)
        if self.target_heads is not None:
            return torch.cat([head(residual) for head in self.target_heads], dim=1)
        assert self.head is not None
        return self.head(residual)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device() -> tuple[str, str]:
    forced = os.getenv("PEPTIDE_DEVICE", "auto").strip().lower()
    if forced not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("PEPTIDE_DEVICE must be one of: auto, cpu, cuda, mps")

    if forced == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PEPTIDE_DEVICE=cuda was requested but CUDA is not available")
        return "cuda", "forced by PEPTIDE_DEVICE=cuda"

    if forced == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("PEPTIDE_DEVICE=mps was requested but MPS is not available")
        return "mps", "forced by PEPTIDE_DEVICE=mps"

    if forced == "cpu":
        return "cpu", "forced by PEPTIDE_DEVICE=cpu"

    system = platform.system()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if system == "Windows" and torch.cuda.is_available():
        return "cuda", "auto-selected CUDA on Windows"
    if system == "Darwin":
        return "cpu", "auto-selected optimized CPU on macOS"
    if torch.cuda.is_available():
        return "cuda", "auto-selected CUDA"
    if has_mps:
        return "mps", "auto-selected MPS"
    return "cpu", "auto-selected CPU fallback"


def configure_runtime() -> RuntimeConfig:
    device_type, selection_reason = resolve_device()
    cpu_count = os.cpu_count() or 1

    default_cpu_threads = max(1, cpu_count - 1) if platform.system() == "Darwin" else max(1, cpu_count)
    default_interop_threads = min(4, max(1, cpu_count // 2))
    cpu_threads = int(os.getenv("PEPTIDE_CPU_THREADS", str(default_cpu_threads)))
    interop_threads = int(os.getenv("PEPTIDE_INTEROP_THREADS", str(default_interop_threads)))

    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(max(1, cpu_threads))
    torch.set_num_interop_threads(max(1, interop_threads))

    if device_type == "cuda":
        torch.backends.cudnn.benchmark = True
        use_amp = True
        amp_dtype = "float16"
        pin_memory = True
        non_blocking = True
    else:
        use_amp = False
        amp_dtype = None
        pin_memory = False
        non_blocking = False

    default_workers = min(8, max(1, cpu_count // 2))
    num_workers = int(os.getenv("PEPTIDE_NUM_WORKERS", str(default_workers)))
    if device_type == "mps":
        num_workers = min(num_workers, 4)

    compile_requested = os.getenv("PEPTIDE_COMPILE", "0").strip().lower() in {"1", "true", "yes"}
    compile_mode = os.getenv("PEPTIDE_COMPILE_MODE", "reduce-overhead") if compile_requested else None

    return RuntimeConfig(
        device_type=device_type,
        device_name=str(torch.device(device_type)),
        selection_reason=selection_reason,
        cpu_threads=max(1, cpu_threads),
        interop_threads=max(1, interop_threads),
        num_workers=max(0, num_workers),
        pin_memory=pin_memory,
        persistent_workers=max(0, num_workers) > 0,
        prefetch_factor=2 if max(0, num_workers) > 0 else None,
        non_blocking=non_blocking,
        use_amp=use_amp,
        amp_dtype=amp_dtype,
        compile_model=compile_requested,
        compile_mode=compile_mode,
    )


def create_loader(dataset: StepDataset, batch_size: int, *, shuffle: bool, runtime: RuntimeConfig) -> DataLoader:
    loader_kwargs: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": runtime.num_workers,
        "pin_memory": runtime.pin_memory,
    }
    if runtime.num_workers > 0:
        loader_kwargs["persistent_workers"] = runtime.persistent_workers
        loader_kwargs["prefetch_factor"] = runtime.prefetch_factor
    return DataLoader(**loader_kwargs)


def move_input_batch(batch: dict[str, torch.Tensor], runtime: RuntimeConfig) -> dict[str, torch.Tensor]:
    device = torch.device(runtime.device_type)
    return {
        key: value.to(device, non_blocking=runtime.non_blocking)
        for key, value in batch.items()
        if key in {"sequence_tokens", "next_tokens", "coupling_tokens", "numeric"}
    }


def move_targets(batch: dict[str, torch.Tensor], runtime: RuntimeConfig) -> torch.Tensor:
    return batch["targets"].to(torch.device(runtime.device_type), non_blocking=runtime.non_blocking)


def autocast_context(runtime: RuntimeConfig) -> contextlib.AbstractContextManager[None]:
    if not runtime.use_amp:
        return contextlib.nullcontext()

    dtype = torch.float16 if runtime.amp_dtype == "float16" else torch.bfloat16
    return torch.amp.autocast(device_type=runtime.device_type, dtype=dtype, enabled=True)


def grouped_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = random.Random(SPLIT_SEED)
    serials = df["serial"].drop_duplicates().tolist()
    rng.shuffle(serials)

    total = len(serials)
    train_serials = set(serials[: int(total * 0.7)])
    val_serials = set(serials[int(total * 0.7): int(total * 0.85)])
    test_serials = set(serials[int(total * 0.85):])

    train_df = df[df["serial"].isin(train_serials)].copy()
    val_df = df[df["serial"].isin(val_serials)].copy()
    test_df = df[df["serial"].isin(test_serials)].copy()
    return train_df, val_df, test_df


def inverse_targets(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std) + mean


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> dict[str, dict[str, float] | float]:
    result: dict[str, dict[str, float] | float] = {}
    rmses = []
    maes = []

    for index, column in enumerate(TARGET_COLUMNS):
        pred = predictions[:, index]
        true = targets[:, index]
        error = pred - true
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(np.square(error))))
        denominator = float(np.sum(np.square(true - np.mean(true))))
        r2 = 1.0 if denominator == 0 else 1.0 - (float(np.sum(np.square(error))) / denominator)
        result[column] = {"mae": mae, "rmse": rmse, "r2": r2}
        maes.append(mae)
        rmses.append(rmse)

    result["combinedMae"] = float(np.mean(maes))
    result["combinedRmse"] = float(np.mean(rmses))
    return result


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    runtime: RuntimeConfig,
    *,
    predict_delta: bool = False,
) -> dict[str, dict[str, float] | float]:
    model.eval()
    predictions = []
    targets = []
    prev_targets = []

    with torch.no_grad():
        for batch in loader:
            input_batch = move_input_batch(batch, runtime)
            with autocast_context(runtime):
                outputs = model(input_batch)
            outputs = outputs.float().cpu().numpy()
            predictions.append(outputs)
            targets.append(batch["raw_targets"].cpu().numpy())
            prev_targets.append(batch["prev_targets"].cpu().numpy())

    normalized_predictions = np.concatenate(predictions, axis=0)
    raw_predictions = inverse_targets(normalized_predictions, target_mean, target_std)
    if predict_delta:
        raw_predictions = raw_predictions + np.concatenate(prev_targets, axis=0)
    raw_targets = np.concatenate(targets, axis=0)
    return compute_metrics(raw_predictions, raw_targets)


def parse_huber_deltas(raw_value: str) -> list[float]:
    if not raw_value.strip():
        return [HUBER_DELTA]
    values = []
    for part in raw_value.split(","):
        value = float(part.strip())
        if value <= 0:
            raise ValueError("Huber delta must be positive")
        values.append(value)
    return values


def build_candidate_configs(huber_deltas: list[float]) -> list[ModelConfig]:
    base_configs = [
        {
            "name": "gru_residual_small",
            "architecture": "gru",
            "max_length": 40,
            "embed_dim": 16,
            "sequence_hidden": 32,
            "numeric_hidden": 32,
            "trunk_hidden": 96,
            "dropout": 0.12,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 30,
            "patience": 5,
        },
        {
            "name": "cnn_residual_medium",
            "architecture": "cnn",
            "max_length": 40,
            "embed_dim": 24,
            "sequence_hidden": 48,
            "numeric_hidden": 32,
            "trunk_hidden": 128,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "weight_decay": 1e-4,
            "batch_size": 256,
            "max_epochs": 32,
            "patience": 6,
        },
        {
            "name": "hybrid_residual_medium",
            "architecture": "hybrid",
            "max_length": 40,
            "embed_dim": 24,
            "sequence_hidden": 48,
            "numeric_hidden": 48,
            "trunk_hidden": 128,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "weight_decay": 2e-4,
            "batch_size": 256,
            "max_epochs": 34,
            "patience": 6,
        },
        {
            "name": "rnn_attention_residual_medium",
            "architecture": "rnn_attention",
            "max_length": 40,
            "embed_dim": 24,
            "sequence_hidden": 40,
            "numeric_hidden": 48,
            "trunk_hidden": 128,
            "dropout": 0.15,
            "learning_rate": 8e-4,
            "weight_decay": 2e-4,
            "batch_size": 256,
            "max_epochs": 34,
            "patience": 6,
        },
        {
            "name": "conditional_gru_attention_delta_heads",
            "architecture": "conditional_gru_attention",
            "max_length": 40,
            "embed_dim": 24,
            "sequence_hidden": 40,
            "numeric_hidden": 48,
            "trunk_hidden": 128,
            "dropout": 0.18,
            "learning_rate": 8e-4,
            "weight_decay": 2e-4,
            "batch_size": 256,
            "max_epochs": 34,
            "patience": 6,
            "include_prev_targets": True,
            "predict_delta": True,
        },
    ]
    configs: list[ModelConfig] = []
    for base in base_configs:
        for delta in huber_deltas:
            delta_tag = str(delta).replace(".", "_")
            config_kwargs = {**base, "name": f"{base['name']}_delta_{delta_tag}", "huber_delta": delta}
            configs.append(
                ModelConfig(**config_kwargs)
            )
    return configs


def candidate_seed_offset(config: ModelConfig) -> int:
    order = {"gru": 0, "cnn": 1, "hybrid": 2, "rnn_attention": 3, "conditional_gru_attention": 4}
    return order[config.architecture]


def build_step_datasets_for_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: ModelConfig,
) -> tuple[StepDataset, StepDataset, StepDataset]:
    train_dataset = StepDataset(
        train_df,
        max_length=config.max_length,
        include_prev_targets=config.include_prev_targets,
        predict_delta=config.predict_delta,
        fit=True,
    )
    val_dataset = StepDataset(
        val_df,
        max_length=config.max_length,
        numeric_mean=train_dataset.numeric_mean,
        numeric_std=train_dataset.numeric_std,
        target_mean=train_dataset.target_mean,
        target_std=train_dataset.target_std,
        include_prev_targets=config.include_prev_targets,
        predict_delta=config.predict_delta,
    )
    test_dataset = StepDataset(
        test_df,
        max_length=config.max_length,
        numeric_mean=train_dataset.numeric_mean,
        numeric_std=train_dataset.numeric_std,
        target_mean=train_dataset.target_mean,
        target_std=train_dataset.target_std,
        include_prev_targets=config.include_prev_targets,
        predict_delta=config.predict_delta,
    )
    return train_dataset, val_dataset, test_dataset


def train_candidate(
    config: ModelConfig,
    train_dataset: StepDataset,
    val_dataset: StepDataset,
    test_dataset: StepDataset,
    runtime: RuntimeConfig,
    seed_offset: int,
    seed_base: int = TRAINING_SEED,
) -> dict[str, object]:
    set_seed(int(seed_base) + seed_offset)

    model = ResidualRegressor(
        config,
        numeric_dim=train_dataset.numeric.shape[1],
        output_dim=len(TARGET_COLUMNS),
    ).to(torch.device(runtime.device_type))
    if runtime.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model, mode=runtime.compile_mode)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = build_torch_huber_loss(config.huber_delta)
    scaler = torch.amp.GradScaler(device="cuda", enabled=runtime.use_amp and runtime.device_type == "cuda")

    train_loader = create_loader(train_dataset, config.batch_size, shuffle=True, runtime=runtime)
    val_loader = create_loader(val_dataset, config.batch_size, shuffle=False, runtime=runtime)
    test_loader = create_loader(test_dataset, config.batch_size, shuffle=False, runtime=runtime)

    best_state = None
    best_epoch = 0
    best_score = float("inf")
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch in train_loader:
            input_batch = move_input_batch(batch, runtime)
            targets = move_targets(batch, runtime)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                predictions = model(input_batch)
                loss = loss_fn(predictions, targets)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

        val_metrics = evaluate_model(
            model,
            val_loader,
            train_dataset.target_mean,
            train_dataset.target_std,
            runtime,
            predict_delta=config.predict_delta,
        )

        if val_metrics["combinedRmse"] + 1e-5 < best_score:
            best_score = float(val_metrics["combinedRmse"])
            best_epoch = epoch
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    assert best_state is not None
    model.load_state_dict(best_state)
    final_val = evaluate_model(
        model,
        val_loader,
        train_dataset.target_mean,
        train_dataset.target_std,
        runtime,
        predict_delta=config.predict_delta,
    )
    final_test = evaluate_model(
        model,
        test_loader,
        train_dataset.target_mean,
        train_dataset.target_std,
        runtime,
        predict_delta=config.predict_delta,
    )

    return {
        "config": asdict(config),
        "loss": {"name": "HuberLoss", "delta": config.huber_delta},
        "bestEpoch": best_epoch,
        "stateDict": best_state,
        "validation": final_val,
        "test": final_test,
    }


def state_dict_to_serializable(state_dict: dict[str, torch.Tensor]) -> dict[str, list]:
    return {key: value.tolist() for key, value in state_dict.items()}


def write_artifacts(
    best_result: dict[str, object],
    *,
    all_results: list[dict[str, object]],
    huber_deltas: list[float],
    train_dataset: StepDataset,
    split_sizes: dict[str, int],
    serial_split_sizes: dict[str, int],
    raw_df: pd.DataFrame,
    runtime: RuntimeConfig,
) -> None:
    artifacts_dir = project_root() / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "real-synthesis-model.pt"
    report_path = artifacts_dir / "real-synthesis-report.json"

    payload = {
        "version": "real-fastflow-multitask-v1",
        "source": {
            "name": REAL_DATASET_SOURCE,
            "url": REAL_DATASET_URL,
        },
        "targets": TARGET_COLUMNS,
        "numericFeatures": [*NUMERIC_COLUMNS, "sequence_length"],
        "categoricalFeatures": CATEGORICAL_COLUMNS,
        "tokenAlphabet": TOKEN_ALPHABET,
        "splitStrategy": "grouped-by-serial",
        "splitSizes": split_sizes,
        "serialSplitSizes": serial_split_sizes,
        "datasetStats": {
            "rows": int(len(raw_df)),
            "uniqueSerials": int(raw_df["serial"].nunique()),
            "maxPreChainLength": int(raw_df["pre-chain"].str.len().max()),
            "meanPreChainLength": float(raw_df["pre-chain"].str.len().mean()),
        },
        "huberDeltaSearch": {
            "candidateDeltas": [float(value) for value in huber_deltas],
            "candidateCount": int(len(all_results)),
        },
        "normalization": {
            "numericMean": train_dataset.numeric_mean.tolist(),
            "numericStd": train_dataset.numeric_std.tolist(),
            "targetMean": train_dataset.target_mean.tolist(),
            "targetStd": train_dataset.target_std.tolist(),
        },
        "runtime": asdict(runtime),
        "bestConfig": best_result["config"],
        "bestLoss": best_result["loss"],
        "bestEpoch": best_result["bestEpoch"],
        "validation": best_result["validation"],
        "test": best_result["test"],
        "candidateResults": [
            {
                "name": item["config"]["name"],
                "architecture": item["config"]["architecture"],
                "huberDelta": item["config"]["huber_delta"],
                "bestEpoch": item["bestEpoch"],
                "validationCombinedRmse": item["validation"]["combinedRmse"],
                "testCombinedRmse": item["test"]["combinedRmse"],
            }
            for item in all_results
        ],
        "stateDict": state_dict_to_serializable(best_result["stateDict"]),
    }

    torch.save(
        {
            "config": best_result["config"],
            "state_dict": best_result["stateDict"],
            "normalization": payload["normalization"],
            "token_alphabet": TOKEN_ALPHABET,
            "targets": TARGET_COLUMNS,
        },
        model_path,
    )
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    runtime = configure_runtime()
    print(
        "Runtime:",
        json.dumps(
            {
                "device": runtime.device_name,
                "selectionReason": runtime.selection_reason,
                "cpuThreads": runtime.cpu_threads,
                "interopThreads": runtime.interop_threads,
                "numWorkers": runtime.num_workers,
                "useAmp": runtime.use_amp,
                "compileModel": runtime.compile_model,
            },
            indent=2,
        ),
    )

    raw_df = load_real_dataset(force_download=False)
    train_df, val_df, test_df = grouped_split(raw_df)
    huber_deltas = parse_huber_deltas(os.getenv("PEPTIDE_HUBER_DELTAS", str(HUBER_DELTA)))
    candidate_configs = build_candidate_configs(huber_deltas)

    results = []
    for index, config in enumerate(candidate_configs):
        train_dataset, val_dataset, test_dataset = build_step_datasets_for_config(train_df, val_df, test_df, config)

        result = train_candidate(
            config,
            train_dataset,
            val_dataset,
            test_dataset,
            runtime,
            seed_offset=candidate_seed_offset(config),
        )
        result["trainDataset"] = train_dataset
        results.append(result)
        print(
            f"[{config.name}] val={result['validation']['combinedRmse']:.4f} "
            f"test={result['test']['combinedRmse']:.4f} "
            f"delta={config.huber_delta:.3f}"
        )

    results.sort(key=lambda item: item["validation"]["combinedRmse"])
    best = results[0]
    train_dataset = best["trainDataset"]

    write_artifacts(
        best,
        all_results=results,
        huber_deltas=huber_deltas,
        train_dataset=train_dataset,
        split_sizes={
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        serial_split_sizes={
            "train": int(train_df["serial"].nunique()),
            "validation": int(val_df["serial"].nunique()),
            "test": int(test_df["serial"].nunique()),
        },
        raw_df=raw_df,
        runtime=runtime,
    )

    print("Best architecture:", best["config"]["name"])
    print("Validation:", json.dumps(best["validation"], indent=2))
    print("Test:", json.dumps(best["test"], indent=2))


if __name__ == "__main__":
    main()
