from __future__ import annotations

import argparse
import json
import os
import platform

import numpy as np
import torch

from real_data import TARGET_COLUMNS, TOKEN_TO_INDEX, encode_token_sequence
from train_real_model import ModelConfig, PREVIOUS_TARGET_COLUMNS, ResidualRegressor


def build_numeric_values(
    config: ModelConfig,
    *,
    pre_chain: str,
    coupling_strokes: float,
    deprotection_strokes: float,
    flow_rate: float,
    temp_coupling: float,
    temp_reactor_1: float,
    prev_targets: np.ndarray,
) -> np.ndarray:
    values = np.array(
        [
            coupling_strokes,
            deprotection_strokes,
            flow_rate,
            temp_coupling,
            temp_reactor_1,
            len(pre_chain),
        ],
        dtype=np.float32,
    )
    if config.include_prev_targets:
        values = np.concatenate([values, prev_targets.astype(np.float32)], axis=0)
    return values


def restore_raw_prediction(
    config: ModelConfig,
    normalized_prediction: np.ndarray,
    target_mean: np.ndarray,
    target_std: np.ndarray,
    prev_targets: np.ndarray,
) -> np.ndarray:
    raw_prediction = (normalized_prediction * target_std) + target_mean
    if config.predict_delta:
        raw_prediction = raw_prediction + prev_targets
    return raw_prediction


def load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location=resolve_inference_device())


def resolve_inference_device() -> torch.device:
    forced = torch.device("cpu")
    env_value = str(os.getenv("PEPTIDE_DEVICE", "auto")).strip().lower()
    if env_value == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if env_value == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if env_value == "cpu":
        return forced

    system = platform.system()
    if system == "Windows" and torch.cuda.is_available():
        return torch.device("cuda")
    if system != "Darwin" and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return forced


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict fast-flow peptide synthesis step metrics from the real-data model.")
    parser.add_argument("--pre-chain", required=True, help="Existing pre-chain sequence in single-letter code.")
    parser.add_argument("--amino-acid", required=True, help="Incoming amino acid in single-letter code.")
    parser.add_argument("--coupling-agent", default="HATU", choices=["HATU", "PYAOP", "DIC"])
    parser.add_argument("--coupling-strokes", type=float, default=7)
    parser.add_argument("--deprotection-strokes", type=float, default=13)
    parser.add_argument("--flow-rate", type=float, default=80000)
    parser.add_argument("--temp-coupling", type=float, default=25.0)
    parser.add_argument("--temp-reactor-1", type=float, default=90.0)
    parser.add_argument("--prev-area", type=float, default=0.0)
    parser.add_argument("--prev-height", type=float, default=0.0)
    parser.add_argument("--prev-width", type=float, default=0.0)
    parser.add_argument("--prev-diff", type=float, default=0.0)
    parser.add_argument("--model-path", default="models/deploy/final-deploy-model.pt")
    args = parser.parse_args()

    device = resolve_inference_device()
    checkpoint = load_checkpoint(args.model_path)
    config = ModelConfig(**checkpoint["config"])
    model = ResidualRegressor(
        config,
        numeric_dim=len(checkpoint["normalization"]["numericMean"]),
        output_dim=len(TARGET_COLUMNS),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    normalization = checkpoint["normalization"]
    prev_targets = np.array(
        [args.prev_area, args.prev_height, args.prev_width, args.prev_diff],
        dtype=np.float32,
    )
    numeric_values = build_numeric_values(
        config,
        pre_chain=args.pre_chain,
        coupling_strokes=args.coupling_strokes,
        deprotection_strokes=args.deprotection_strokes,
        flow_rate=args.flow_rate,
        temp_coupling=args.temp_coupling,
        temp_reactor_1=args.temp_reactor_1,
        prev_targets=prev_targets,
    )
    numeric_mean = np.array(normalization["numericMean"], dtype=np.float32)
    numeric_std = np.array(normalization["numericStd"], dtype=np.float32)
    target_mean = np.array(normalization["targetMean"], dtype=np.float32)
    target_std = np.array(normalization["targetStd"], dtype=np.float32)

    normalized_numeric = np.asarray((numeric_values - numeric_mean) / numeric_std, dtype=np.float32)[None, :]

    batch = {
        "sequence_tokens": torch.tensor([encode_token_sequence(args.pre_chain, config.max_length)], dtype=torch.long, device=device),
        "next_tokens": torch.tensor([TOKEN_TO_INDEX.get(args.amino_acid, 0)], dtype=torch.long, device=device),
        "coupling_tokens": torch.tensor([0 if args.coupling_agent == "HATU" else 1], dtype=torch.long, device=device),
        "numeric": torch.tensor(normalized_numeric, dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        prediction = model(batch).cpu().numpy()[0]

    raw_prediction = restore_raw_prediction(config, prediction, target_mean, target_std, prev_targets)
    payload = {name: float(raw_prediction[index]) for index, name in enumerate(TARGET_COLUMNS)}
    payload["model"] = config.name
    payload["inputs"] = {
        "pre_chain": args.pre_chain,
        "amino_acid": args.amino_acid,
        "coupling_agent": args.coupling_agent,
    }
    if config.include_prev_targets:
        payload["previousTargets"] = {
            name: float(value) for name, value in zip(PREVIOUS_TARGET_COLUMNS, prev_targets, strict=True)
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
