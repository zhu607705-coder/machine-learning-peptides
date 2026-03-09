from __future__ import annotations

import os
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from losses import DEFAULT_HUBER_DELTA, huber_loss_and_gradient
from peptide_core import AMINO_ACIDS, FEATURE_NAMES, PeptideSynthesisParams, extract_feature_bundle


ScenarioName = str

DATASET_SEED = 20260301
SPLIT_SEED = 20260302
TRAINING_SEED = 20260303
HUBER_DELTA = DEFAULT_HUBER_DELTA

BULKY_POOL = ["Val", "Ile", "Thr", "Pro", "Phe", "Trp", "Tyr", "Leu"]
HYDROPHOBIC_POOL = ["Ala", "Val", "Ile", "Leu", "Phe", "Trp", "Tyr", "Met"]
CHARGED_POOL = ["Arg", "His", "Lys", "Asp", "Glu"]
DIFFICULT_POOL = ["Arg", "Cys", "His", "Asn", "Gln"]


@dataclass(frozen=True)
class WeightedOption:
    value: str
    weight: float


@dataclass(frozen=True)
class ScenarioConfig:
    name: ScenarioName
    sample_count: int
    length_range: Tuple[int, int]
    amino_weights: List[WeightedOption]
    topology_weights: List[WeightedOption]
    reagent_weights: List[WeightedOption]
    solvent_weights: List[WeightedOption]
    temperature_weights: List[WeightedOption]
    cleavage_weights: List[WeightedOption]
    purity_bias: float
    yield_bias: float
    noise_scale: float


@dataclass(frozen=True)
class ExampleRecord:
    scenario: ScenarioName
    params: PeptideSynthesisParams
    features: List[float]
    targets: Tuple[float, float]


@dataclass(frozen=True)
class CandidateConfig:
    hidden_size: int
    learning_rate: float
    l2: float
    max_epochs: int
    patience: int


@dataclass(frozen=True)
class PreparedDataset:
    features: np.ndarray
    targets: np.ndarray


def clamp(value: float, min_value: float, max_value: float) -> float:
    return min(max_value, max(min_value, value))


def weight_map(default_weight: float, overrides: Dict[str, float]) -> List[WeightedOption]:
    return [WeightedOption(value=amino_acid, weight=overrides.get(amino_acid, default_weight)) for amino_acid in AMINO_ACIDS]


def sample_weighted(rng: random.Random, options: Sequence[WeightedOption]) -> str:
    total = sum(option.weight for option in options)
    cursor = rng.random() * total
    for option in options:
        cursor -= option.weight
        if cursor <= 0:
            return option.value
    return options[-1].value


def random_residue_from_pool(rng: random.Random, pool: Sequence[str]) -> str:
    return pool[rng.randrange(0, len(pool))]


def ensure_residue_count(rng: random.Random, residues: List[str], pool: Sequence[str], minimum_count: int) -> None:
    count = sum(1 for residue in residues if residue in pool)
    while count < minimum_count:
      replace_index = rng.randrange(0, len(residues))
      residues[replace_index] = random_residue_from_pool(rng, pool)
      count = sum(1 for residue in residues if residue in pool)


def ensure_hydrophobic_run(rng: random.Random, residues: List[str], minimum_run: int) -> None:
    if minimum_run <= 1 or len(residues) < minimum_run:
        return
    start = rng.randrange(0, len(residues) - minimum_run + 1)
    for offset in range(minimum_run):
        residues[start + offset] = random_residue_from_pool(rng, HYDROPHOBIC_POOL)


SCENARIOS = [
    ScenarioConfig(
        name="easy_linear",
        sample_count=180,
        length_range=(4, 8),
        amino_weights=weight_map(1, {"Gly": 3.2, "Ala": 3.0, "Ser": 2.2, "Leu": 1.8, "Gln": 1.6, "Cys": 0.4, "Arg": 0.6, "Trp": 0.6}),
        topology_weights=[WeightedOption("Linear", 1)],
        reagent_weights=[WeightedOption("HATU", 5), WeightedOption("PyBOP", 3), WeightedOption("HBTU", 2), WeightedOption("DIC/Oxyma", 2)],
        solvent_weights=[WeightedOption("DMF", 5), WeightedOption("NMP", 3), WeightedOption("DMF/DCM", 1)],
        temperature_weights=[WeightedOption("Room Temperature", 6), WeightedOption("Microwave 75°C", 2), WeightedOption("Microwave 90°C", 1)],
        cleavage_weights=[WeightedOption("2 hours", 4), WeightedOption("3 hours", 4), WeightedOption("4 hours", 1)],
        purity_bias=2.2,
        yield_bias=1.8,
        noise_scale=1.2,
    ),
    ScenarioConfig(
        name="hydrophobic_long",
        sample_count=180,
        length_range=(8, 16),
        amino_weights=weight_map(0.9, {"Ala": 2.1, "Val": 3.0, "Ile": 2.8, "Leu": 2.8, "Phe": 2.4, "Trp": 1.8, "Tyr": 1.6, "Met": 1.8, "Gly": 0.6, "Asp": 0.4}),
        topology_weights=[WeightedOption("Linear", 8), WeightedOption("Head-to-tail cyclization", 2)],
        reagent_weights=[WeightedOption("HATU", 4), WeightedOption("PyBOP", 3), WeightedOption("HBTU", 2), WeightedOption("DIC/Oxyma", 1)],
        solvent_weights=[WeightedOption("DMF/DCM", 5), WeightedOption("NMP", 3), WeightedOption("DMF", 2)],
        temperature_weights=[WeightedOption("Microwave 75°C", 5), WeightedOption("Room Temperature", 3), WeightedOption("Microwave 90°C", 2)],
        cleavage_weights=[WeightedOption("3 hours", 4), WeightedOption("4 hours", 4), WeightedOption("2 hours", 1)],
        purity_bias=-2.4,
        yield_bias=-2.8,
        noise_scale=1.8,
    ),
    ScenarioConfig(
        name="steric_rich",
        sample_count=180,
        length_range=(7, 12),
        amino_weights=weight_map(0.8, {"Val": 2.8, "Ile": 2.8, "Thr": 2.6, "Pro": 2.4, "Phe": 2.0, "Tyr": 1.8, "Gly": 0.6, "Asp": 0.5, "Arg": 0.6}),
        topology_weights=[WeightedOption("Linear", 8), WeightedOption("Head-to-tail cyclization", 1)],
        reagent_weights=[WeightedOption("HATU", 4), WeightedOption("PyBOP", 3), WeightedOption("HBTU", 2), WeightedOption("DIC/Oxyma", 1)],
        solvent_weights=[WeightedOption("NMP", 4), WeightedOption("DMF/DCM", 3), WeightedOption("DMF", 3)],
        temperature_weights=[WeightedOption("Microwave 75°C", 5), WeightedOption("Room Temperature", 3), WeightedOption("Microwave 90°C", 2)],
        cleavage_weights=[WeightedOption("3 hours", 5), WeightedOption("4 hours", 2), WeightedOption("2 hours", 2)],
        purity_bias=-1.6,
        yield_bias=-1.2,
        noise_scale=1.6,
    ),
    ScenarioConfig(
        name="charged_difficult",
        sample_count=180,
        length_range=(7, 14),
        amino_weights=weight_map(0.9, {"Arg": 2.8, "His": 2.0, "Lys": 1.8, "Asp": 2.4, "Glu": 2.2, "Asn": 2.0, "Cys": 1.4, "Gln": 1.8, "Val": 0.6}),
        topology_weights=[WeightedOption("Linear", 7), WeightedOption("Disulfide cyclization", 1), WeightedOption("Head-to-tail cyclization", 1)],
        reagent_weights=[WeightedOption("HATU", 4), WeightedOption("PyBOP", 2), WeightedOption("HBTU", 3), WeightedOption("DIC/Oxyma", 1)],
        solvent_weights=[WeightedOption("NMP", 4), WeightedOption("DMF", 4), WeightedOption("DMF/DCM", 2)],
        temperature_weights=[WeightedOption("Room Temperature", 5), WeightedOption("Microwave 75°C", 3), WeightedOption("Microwave 90°C", 1)],
        cleavage_weights=[WeightedOption("3 hours", 4), WeightedOption("4 hours", 4), WeightedOption("2 hours", 1)],
        purity_bias=-2.0,
        yield_bias=-1.8,
        noise_scale=2.1,
    ),
    ScenarioConfig(
        name="cysteine_rich",
        sample_count=180,
        length_range=(6, 12),
        amino_weights=weight_map(0.8, {"Cys": 3.2, "Gly": 1.8, "Ser": 1.8, "Ala": 1.4, "Arg": 1.2, "His": 1.2, "Asp": 0.8}),
        topology_weights=[WeightedOption("Disulfide cyclization", 6), WeightedOption("Linear", 4)],
        reagent_weights=[WeightedOption("HATU", 5), WeightedOption("PyBOP", 2), WeightedOption("HBTU", 2), WeightedOption("DIC/Oxyma", 1)],
        solvent_weights=[WeightedOption("DMF", 4), WeightedOption("NMP", 4), WeightedOption("DMF/DCM", 2)],
        temperature_weights=[WeightedOption("Room Temperature", 6), WeightedOption("Microwave 75°C", 3), WeightedOption("Microwave 90°C", 1)],
        cleavage_weights=[WeightedOption("3 hours", 4), WeightedOption("4 hours", 4), WeightedOption("2 hours", 2)],
        purity_bias=-2.4,
        yield_bias=-2.0,
        noise_scale=2.4,
    ),
    ScenarioConfig(
        name="cyclized",
        sample_count=180,
        length_range=(5, 10),
        amino_weights=weight_map(0.9, {"Gly": 1.8, "Ala": 1.4, "Val": 1.8, "Pro": 1.8, "Cys": 1.6, "Asp": 1.4, "Lys": 1.2, "Phe": 1.4}),
        topology_weights=[WeightedOption("Head-to-tail cyclization", 6), WeightedOption("Disulfide cyclization", 3), WeightedOption("Linear", 1)],
        reagent_weights=[WeightedOption("HATU", 5), WeightedOption("PyBOP", 3), WeightedOption("HBTU", 1), WeightedOption("DIC/Oxyma", 1)],
        solvent_weights=[WeightedOption("NMP", 5), WeightedOption("DMF/DCM", 3), WeightedOption("DMF", 2)],
        temperature_weights=[WeightedOption("Room Temperature", 4), WeightedOption("Microwave 75°C", 4), WeightedOption("Microwave 90°C", 2)],
        cleavage_weights=[WeightedOption("4 hours", 5), WeightedOption("3 hours", 3), WeightedOption("2 hours", 1)],
        purity_bias=-2.8,
        yield_bias=-3.2,
        noise_scale=2.0,
    ),
]


def build_sequence(rng: random.Random, scenario: ScenarioConfig, topology: str) -> str:
    length = rng.randint(*scenario.length_range)
    residues = [sample_weighted(rng, scenario.amino_weights) for _ in range(length)]

    if scenario.name == "hydrophobic_long":
        ensure_hydrophobic_run(rng, residues, min(4, len(residues)))
    if scenario.name == "steric_rich":
        ensure_residue_count(rng, residues, BULKY_POOL, max(2, math.ceil(length * 0.35)))
    if scenario.name == "charged_difficult":
        ensure_residue_count(rng, residues, CHARGED_POOL, max(2, math.ceil(length * 0.3)))
        ensure_residue_count(rng, residues, DIFFICULT_POOL, max(2, math.ceil(length * 0.25)))
    if scenario.name == "cysteine_rich" or "Disulfide" in topology:
        ensure_residue_count(rng, residues, ["Cys"], 2)
    if scenario.name == "cyclized" and "Head-to-tail" in topology and length < 6:
        residues.append(random_residue_from_pool(rng, ["Gly", "Ala", "Val", "Pro"]))

    c_terminal = "NH2" if rng.random() > 0.82 else "OH"
    return f"H-{'-'.join(residues)}-{c_terminal}"


def simulate_targets(rng: random.Random, scenario: ScenarioConfig, params: PeptideSynthesisParams) -> Tuple[float, float]:
    summary = extract_feature_bundle(params).summary
    weak_reagent = summary.reagent_score < 0.62
    strong_reagent = summary.reagent_score > 0.9
    strong_solvent = summary.solvent_score > 0.8
    high_temperature = summary.temperature_score > 0.95
    moderate_microwave = "75" in params.temperature
    disulfide_mismatch = "Disulfide" in params.topology and summary.cys_count < 2
    short_head_to_tail = "Head-to-tail" in params.topology and summary.length < 6

    purity = (
        91.5
        - (16 * summary.length_norm)
        - (11 * summary.bulky_ratio)
        - (13.5 * summary.difficult_ratio)
        - (9.5 * summary.hydrophobic_ratio)
        - (8 * summary.topology_complexity)
        - (7 * summary.aspartimide_risk)
        + (12 * summary.reagent_score)
        + (5 * summary.solvent_score)
        + (4.5 * summary.temperature_score)
        + (4.2 * summary.cleavage_score)
        + (5.5 * summary.breaker_ratio)
        + (4.5 * summary.sequence_complexity)
        - (8 * summary.longest_hydrophobic_run)
        + (5 * summary.cys_cyclization_fit)
        + scenario.purity_bias
    )
    if summary.bulky_ratio > 0.3 and strong_reagent:
        purity += 4.2
    if summary.hydrophobic_ratio > 0.45 and strong_solvent:
        purity += 4.8
    if summary.difficult_ratio > 0.28 and weak_reagent:
        purity -= 6.5
    if moderate_microwave and summary.bulky_ratio > 0.28:
        purity += 3.2
    if high_temperature and summary.sensitive_ratio > 0.2:
        purity -= 5.4
    if disulfide_mismatch:
        purity -= 12
    if short_head_to_tail:
        purity -= 8.5
    if "Disulfide" in params.topology and summary.cys_count >= 2 and params.temperature == "Room Temperature":
        purity += 2.6
    if summary.length > 10 and params.cleavage_time == "4 hours":
        purity += 1.4
    purity += rng.gauss(0, scenario.noise_scale)
    purity = clamp(purity, 28, 99.5)

    yield_value = (
        71
        + (0.62 * (purity - 70))
        - (13 * summary.length_norm)
        - (8.5 * summary.topology_complexity)
        - (8 * summary.difficult_ratio)
        - (6.5 * summary.longest_hydrophobic_run)
        + (5.5 * summary.solvent_score)
        + (3.5 * summary.cleavage_score)
        + (2.5 * summary.sequence_complexity)
        + (2 * summary.breaker_ratio)
        + scenario.yield_bias
    )
    if moderate_microwave and summary.bulky_ratio > 0.25:
        yield_value += 3.3
    if high_temperature and (summary.aspartimide_risk + summary.sulfur_ratio) > 0.25:
        yield_value -= 4.8
    if strong_reagent and summary.difficult_ratio > 0.2:
        yield_value += 2.4
    if disulfide_mismatch:
        yield_value -= 10
    if short_head_to_tail:
        yield_value -= 5.5
    yield_value += rng.gauss(0, scenario.noise_scale * 1.15)
    yield_value = clamp(yield_value, 12, 95)
    yield_value = min(yield_value, purity + 6)
    return purity, yield_value


def create_dataset() -> List[ExampleRecord]:
    rng = random.Random(DATASET_SEED)
    examples: List[ExampleRecord] = []
    for scenario in SCENARIOS:
        for _ in range(scenario.sample_count):
            topology = sample_weighted(rng, scenario.topology_weights)
            params = PeptideSynthesisParams(
                sequence=build_sequence(rng, scenario, topology),
                topology=topology,
                coupling_reagent=sample_weighted(rng, scenario.reagent_weights),
                solvent=sample_weighted(rng, scenario.solvent_weights),
                temperature=sample_weighted(rng, scenario.temperature_weights),
                cleavage_time=sample_weighted(rng, scenario.cleavage_weights),
            )
            examples.append(
                ExampleRecord(
                    scenario=scenario.name,
                    params=params,
                    features=extract_feature_bundle(params).vector,
                    targets=simulate_targets(rng, scenario, params),
                )
            )
    return examples


def stratified_split(examples: List[ExampleRecord]) -> Tuple[List[ExampleRecord], List[ExampleRecord], List[ExampleRecord]]:
    rng = random.Random(SPLIT_SEED)
    buckets: Dict[ScenarioName, List[ExampleRecord]] = {}
    for example in examples:
        buckets.setdefault(example.scenario, []).append(example)

    train: List[ExampleRecord] = []
    validation: List[ExampleRecord] = []
    test: List[ExampleRecord] = []

    for scenario in SCENARIOS:
        bucket = list(buckets.get(scenario.name, []))
        rng.shuffle(bucket)
        train_count = int(len(bucket) * 0.7)
        validation_count = int(len(bucket) * 0.15)
        train.extend(bucket[:train_count])
        validation.extend(bucket[train_count:train_count + validation_count])
        test.extend(bucket[train_count + validation_count:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def examples_to_arrays(examples: List[ExampleRecord]) -> PreparedDataset:
    return PreparedDataset(
        features=np.asarray([record.features for record in examples], dtype=np.float64),
        targets=np.asarray([record.targets for record in examples], dtype=np.float64),
    )


def compute_normalization(vectors: np.ndarray) -> Dict[str, List[float]]:
    mean = vectors.mean(axis=0)
    std = vectors.std(axis=0)
    floor = 1e-6 + (np.abs(mean) * 1e-6)
    std = np.maximum(std, floor)
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
    }


def normalize_dataset(
    dataset: PreparedDataset,
    feature_stats: Dict[str, List[float]],
    target_stats: Dict[str, List[float]],
) -> PreparedDataset:
    feature_mean = np.asarray(feature_stats["mean"], dtype=np.float64)
    feature_std = np.asarray(feature_stats["std"], dtype=np.float64)
    target_mean = np.asarray(target_stats["mean"], dtype=np.float64)
    target_std = np.asarray(target_stats["std"], dtype=np.float64)
    return PreparedDataset(
        features=(dataset.features - feature_mean) / feature_std,
        targets=(dataset.targets - target_mean) / target_std,
    )


def create_network(
    input_size: int,
    hidden_size: int,
    output_size: int,
    rng: random.Random,
) -> Dict[str, np.ndarray]:
    limit1 = math.sqrt(6 / (input_size + hidden_size))
    limit2 = math.sqrt(6 / (hidden_size + output_size))
    return {
        "W1": np.asarray(
            [[((rng.random() * 2) - 1) * limit1 for _ in range(input_size)] for _ in range(hidden_size)],
            dtype=np.float64,
        ),
        "b1": np.zeros(hidden_size, dtype=np.float64),
        "W2": np.asarray(
            [[((rng.random() * 2) - 1) * limit2 for _ in range(hidden_size)] for _ in range(output_size)],
            dtype=np.float64,
        ),
        "b2": np.zeros(output_size, dtype=np.float64),
    }


def clone_network(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {
        "W1": weights["W1"].copy(),
        "b1": weights["b1"].copy(),
        "W2": weights["W2"].copy(),
        "b2": weights["b2"].copy(),
    }


def forward_pass(weights: Dict[str, np.ndarray], features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hidden_raw = features @ weights["W1"].T + weights["b1"]
    hidden = np.tanh(hidden_raw)
    outputs = hidden @ weights["W2"].T + weights["b2"]
    return hidden, outputs


def denormalize_targets(values: np.ndarray, target_stats: Dict[str, List[float]]) -> np.ndarray:
    target_mean = np.asarray(target_stats["mean"], dtype=np.float64)
    target_std = np.asarray(target_stats["std"], dtype=np.float64)
    return (values * target_std) + target_mean


def compute_regression_metrics(actual: List[float], predicted: List[float]) -> Dict[str, float]:
    actual_array = np.asarray(actual, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    error = predicted_array - actual_array
    centered = actual_array - actual_array.mean()
    squared_error = float(np.square(error).sum())
    total_squared_error = float(np.square(centered).sum())
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.square(error).mean())),
        "r2": 1.0 if total_squared_error == 0 else 1.0 - (squared_error / total_squared_error),
    }


def evaluate(weights: Dict[str, np.ndarray], dataset: PreparedDataset, target_stats: Dict[str, List[float]]) -> Dict[str, object]:
    _, normalized_predictions = forward_pass(weights, dataset.features)
    raw_predictions = denormalize_targets(normalized_predictions, target_stats)
    raw_targets = denormalize_targets(dataset.targets, target_stats)

    purity = compute_regression_metrics(raw_targets[:, 0].tolist(), raw_predictions[:, 0].tolist())
    yield_metrics = compute_regression_metrics(raw_targets[:, 1].tolist(), raw_predictions[:, 1].tolist())
    return {
        "purity": purity,
        "yield": yield_metrics,
        "combinedMae": (purity["mae"] + yield_metrics["mae"]) / 2.0,
        "combinedRmse": (purity["rmse"] + yield_metrics["rmse"]) / 2.0,
    }


def train_candidate(
    config: CandidateConfig,
    train_set: PreparedDataset,
    validation_set: PreparedDataset,
    test_set: PreparedDataset,
    target_stats: Dict[str, List[float]],
    seed_offset: int,
) -> Dict[str, object]:
    rng = random.Random(TRAINING_SEED + seed_offset)
    input_size = int(train_set.features.shape[1])
    output_size = int(train_set.targets.shape[1])
    weights = create_network(input_size, config.hidden_size, output_size, rng)
    best_weights = clone_network(weights)
    best_epoch = 0
    best_score = float("inf")
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        hidden, output = forward_pass(weights, train_set.features)
        _, output_gradient = huber_loss_and_gradient(output, train_set.targets, delta=HUBER_DELTA)
        d_output = output_gradient / output_size
        grad_w2 = d_output.T @ hidden
        grad_b2 = d_output.sum(axis=0)

        d_hidden = (d_output @ weights["W2"]) * (1.0 - np.square(hidden))
        grad_w1 = d_hidden.T @ train_set.features
        grad_b1 = d_hidden.sum(axis=0)

        learning_rate = config.learning_rate / train_set.features.shape[0]
        weights["b1"] -= learning_rate * grad_b1
        weights["W1"] -= learning_rate * (grad_w1 + (config.l2 * weights["W1"]))
        weights["b2"] -= learning_rate * grad_b2
        weights["W2"] -= learning_rate * (grad_w2 + (config.l2 * weights["W2"]))

        validation_metrics = evaluate(weights, validation_set, target_stats)
        if validation_metrics["combinedRmse"] + 1e-5 < best_score:
            best_score = validation_metrics["combinedRmse"]
            best_weights = clone_network(weights)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= config.patience:
            break

    return {
        "config": {
            "hiddenSize": config.hidden_size,
            "learningRate": config.learning_rate,
            "l2": config.l2,
            "loss": "huber",
            "huberDelta": HUBER_DELTA,
            "maxEpochs": config.max_epochs,
            "patience": config.patience,
        },
        "bestEpoch": best_epoch,
        "weights": {
            "W1": best_weights["W1"].tolist(),
            "b1": best_weights["b1"].tolist(),
            "W2": best_weights["W2"].tolist(),
            "b2": best_weights["b2"].tolist(),
        },
        "validation": evaluate(best_weights, validation_set, target_stats),
        "test": evaluate(best_weights, test_set, target_stats),
    }


def train_candidate_task(args: tuple[CandidateConfig, PreparedDataset, PreparedDataset, PreparedDataset, Dict[str, List[float]], int]) -> Dict[str, object]:
    config, train_set, validation_set, test_set, target_stats, seed_offset = args
    return train_candidate(config, train_set, validation_set, test_set, target_stats, seed_offset)


def determine_parallel_jobs(candidate_count: int) -> int:
    cpu_count = os.cpu_count() or 1
    requested = os.getenv("PEPTIDE_OPTIMIZE_JOBS", "").strip()
    if requested:
        return max(1, min(candidate_count, int(requested)))
    return max(1, min(candidate_count, max(1, cpu_count // 2)))


def write_artifacts(model: Dict[str, object]) -> None:
    root_dir = Path(__file__).resolve().parents[1]
    artifacts_dir = root_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    json_model_path = artifacts_dir / "peptide-model.json"
    report_path = artifacts_dir / "peptide-model-report.json"
    model_ts_path = root_dir / "src" / "lib" / "modelArtifacts.ts"

    json_model_path.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(model["training"], indent=2) + "\n", encoding="utf-8")
    model_ts_path.write_text(
        "import type { TrainedPeptideNetwork } from './neuralModel';\n\n"
        f"export const trainedPeptideNetwork: TrainedPeptideNetwork = {json.dumps(model, indent=2)};\n\n"
        "export default trainedPeptideNetwork;\n",
        encoding="utf-8",
    )


def main() -> None:
    dataset = create_dataset()
    train, validation, test = stratified_split(dataset)
    train_arrays = examples_to_arrays(train)
    validation_arrays = examples_to_arrays(validation)
    test_arrays = examples_to_arrays(test)
    feature_stats = compute_normalization(train_arrays.features)
    target_stats = compute_normalization(train_arrays.targets)
    train_set = normalize_dataset(train_arrays, feature_stats, target_stats)
    validation_set = normalize_dataset(validation_arrays, feature_stats, target_stats)
    test_set = normalize_dataset(test_arrays, feature_stats, target_stats)

    candidates = [
        CandidateConfig(hidden_size=8, learning_rate=0.08, l2=0.0005, max_epochs=420, patience=55),
        CandidateConfig(hidden_size=8, learning_rate=0.12, l2=0.0015, max_epochs=420, patience=55),
        CandidateConfig(hidden_size=12, learning_rate=0.08, l2=0.0005, max_epochs=480, patience=60),
        CandidateConfig(hidden_size=12, learning_rate=0.12, l2=0.0015, max_epochs=480, patience=60),
        CandidateConfig(hidden_size=16, learning_rate=0.06, l2=0.0005, max_epochs=520, patience=65),
        CandidateConfig(hidden_size=16, learning_rate=0.09, l2=0.0015, max_epochs=520, patience=65),
    ]
    parallel_jobs = determine_parallel_jobs(len(candidates))
    task_args = [
        (candidate, train_set, validation_set, test_set, target_stats, index)
        for index, candidate in enumerate(candidates)
    ]
    if parallel_jobs == 1:
        results = [train_candidate_task(task) for task in task_args]
    else:
        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            results = list(executor.map(train_candidate_task, task_args))
    results.sort(key=lambda result: result["validation"]["combinedRmse"])
    best = results[0]

    model = {
        "version": "synthetic-gridsearch-v3",
        "activation": "tanh",
        "featureNames": FEATURE_NAMES,
        "hiddenSize": best["config"]["hiddenSize"],
        "inputNormalization": feature_stats,
        "targetNormalization": target_stats,
        "weights": best["weights"],
        "training": {
            "datasetSeed": DATASET_SEED,
            "splitSeed": SPLIT_SEED,
            "scenarioCounts": {scenario.name: scenario.sample_count for scenario in SCENARIOS},
            "splitSizes": {"train": len(train), "validation": len(validation), "test": len(test)},
            "bestConfig": {**best["config"], "bestEpoch": best["bestEpoch"]},
            "parallelJobs": parallel_jobs,
            "validation": best["validation"],
            "test": best["test"],
        },
    }

    write_artifacts(model)

    print("Synthetic dataset size:", len(dataset))
    print("Train/Validation/Test:", len(train), len(validation), len(test))
    print("Best config:", model["training"]["bestConfig"])
    print(
        "Validation RMSE:",
        {
            "purity": f"{model['training']['validation']['purity']['rmse']:.3f}",
            "yield": f"{model['training']['validation']['yield']['rmse']:.3f}",
            "combined": f"{model['training']['validation']['combinedRmse']:.3f}",
        },
    )
    print(
        "Test RMSE:",
        {
            "purity": f"{model['training']['test']['purity']['rmse']:.3f}",
            "yield": f"{model['training']['test']['yield']['rmse']:.3f}",
            "combined": f"{model['training']['test']['combinedRmse']:.3f}",
        },
    )


if __name__ == "__main__":
    main()
