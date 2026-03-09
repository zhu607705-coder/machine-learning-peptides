from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import joblib
import pandas as pd

from enhance_dataset import AMINO_ACID_PROPERTIES, COUPLING_DIFFICULTY, calculate_peptide_features, parse_sequence
from neural_model import load_model, predict_targets
from peptide_core import AA_MASS, PeptideSynthesisParams, calculate_exact_mass_mh, cleavage_score, extract_feature_bundle, reagent_score, solvent_score, temperature_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATED_MODELS = {
    "purity": PROJECT_ROOT / "models" / "annotated" / "purity_full_sourcerouted.joblib",
    "yield_val": PROJECT_ROOT / "models" / "annotated" / "yield_val_full_sourcerouted.joblib",
}
POLAR_RESIDUES = {"S", "T", "N", "Q", "C", "Y"}
CHARGED_RESIDUES = {"D", "E", "K", "R", "H"}
AROMATIC_RESIDUES = {"F", "W", "Y", "H"}
SULFUR_RESIDUES = {"C", "M"}
_ANNOTATED_MODEL_CACHE: dict[str, object] = {}


def _create_seeded_random(payload: Dict[str, Any]) -> random.Random:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def _first_matching_residue(residues: List[str], target: set[str]) -> str | None:
    for residue in residues:
        if residue in target:
            return residue
    return None


def _ratio_from_residues(residues: list[str], target: set[str]) -> float:
    if not residues:
        return 0.0
    return sum(1 for residue in residues if residue in target) / len(residues)


def build_annotated_feature_row(params: PeptideSynthesisParams, data_source: str = "literature") -> dict[str, float | str]:
    features = calculate_peptide_features(params.sequence)
    residues = parse_sequence(params.sequence)
    return {
        "length": float(features.length),
        "avg_hydrophobicity": float(features.avg_hydrophobicity),
        "total_charge": float(features.total_charge),
        "molecular_weight": float(features.molecular_weight),
        "avg_volume": float(features.avg_volume),
        "max_coupling_difficulty": float(features.max_coupling_difficulty),
        "bulky_ratio": float(_ratio_from_residues(residues, {"V", "I", "T", "P", "F", "W", "Y", "L"})),
        "polar_ratio": float(_ratio_from_residues(residues, POLAR_RESIDUES)),
        "charged_ratio": float(_ratio_from_residues(residues, CHARGED_RESIDUES)),
        "aromatic_ratio": float(_ratio_from_residues(residues, AROMATIC_RESIDUES)),
        "sulfur_ratio": float(_ratio_from_residues(residues, SULFUR_RESIDUES)),
        "reagent_score": float(reagent_score(params.coupling_reagent)),
        "solvent_score": float(solvent_score(params.solvent)),
        "temperature_score": float(temperature_score(params.temperature)),
        "cleavage_score": float(cleavage_score(params.cleavage_time)),
        "data_source": data_source,
    }


def _load_annotated_model(target: str) -> object:
    if target not in _ANNOTATED_MODEL_CACHE:
        _ANNOTATED_MODEL_CACHE[target] = joblib.load(DEFAULT_ANNOTATED_MODELS[target])
    return _ANNOTATED_MODEL_CACHE[target]


def _predict_annotated_targets(params: PeptideSynthesisParams, data_source_hint: str = "literature") -> dict[str, float]:
    row = build_annotated_feature_row(params, data_source=data_source_hint)
    frame = pd.DataFrame([row])
    purity_model = _load_annotated_model("purity")
    yield_model = _load_annotated_model("yield_val")
    purity = float(purity_model.predict(frame)[0])
    predicted_yield = float(yield_model.predict(frame)[0])
    return {
        "purity": min(99.5, max(10.0, purity)),
        "yield": min(100.0, max(0.5, predicted_yield)),
    }


def predict(
    params: PeptideSynthesisParams,
    model_path: Path | None = None,
    model_family: str = "annotated",
    data_source_hint: str = "literature",
) -> Dict[str, Any]:
    model = None
    bundle = extract_feature_bundle(params)
    if model_family == "annotated":
        raw_prediction = _predict_annotated_targets(params, data_source_hint=data_source_hint)
        purity = raw_prediction["purity"]
        predicted_yield = min(95.0, max(5.0, min(raw_prediction["yield"], purity + 6.0)))
        training_info = {
            "modelFamily": "annotated",
            "purityModel": DEFAULT_ANNOTATED_MODELS["purity"].name,
            "yieldModel": DEFAULT_ANNOTATED_MODELS["yield_val"].name,
            "dataSourceHint": data_source_hint,
        }
    else:
        model = load_model(model_path)
        raw_prediction = predict_targets(model, bundle.vector)
        purity = min(99.5, max(10.0, raw_prediction["purity"]))
        predicted_yield = min(95.0, max(5.0, min(raw_prediction["yield"], purity + 6.0)))
        training_info = model["training"]
    exact_mass_mh = calculate_exact_mass_mh(params.sequence, params.topology)
    seeded_random = _create_seeded_random(
        {
            "sequence": params.sequence,
            "topology": params.topology,
            "coupling_reagent": params.coupling_reagent,
            "solvent": params.solvent,
            "temperature": params.temperature,
            "cleavage_time": params.cleavage_time,
        }
    )
    byproducts: List[Dict[str, Any]] = []
    optimizations: List[str] = []
    mass_spectrum: List[Dict[str, Any]] = [
        {"mz": exact_mass_mh, "abundance": 100.0, "label": "[M+H]+ (Target)"},
        {"mz": exact_mass_mh + 21.9819, "abundance": 10.0 + (seeded_random.random() * 10.0), "label": "[M+Na]+"},
        {"mz": (exact_mass_mh / 2.0) + 0.5, "abundance": 24.0 + (seeded_random.random() * 16.0), "label": "[M+2H]2+"},
    ]

    residues = bundle.residues
    summary = bundle.summary
    bulky_residue = _first_matching_residue(residues, {"Val", "Ile", "Thr", "Pro", "Phe", "Trp", "Tyr", "Leu"})

    if summary.bulky_ratio > 0.28 and bulky_residue:
        deletion_mass = exact_mass_mh - AA_MASS.get(bulky_residue, 100.0)
        byproducts.append(
            {
                "name": f"-{bulky_residue} deletion peptide",
                "mz": deletion_mass,
                "cause": f"Steric crowding around {bulky_residue} can leave a truncated sequence after incomplete coupling.",
            }
        )
        mass_spectrum.append(
            {
                "mz": deletion_mass,
                "abundance": max(6.0, 42.0 - (purity * 0.28)),
                "label": f"[M-{bulky_residue}+H]+",
            }
        )

    if summary.aspartimide_risk > 0.12:
        byproducts.append(
            {
                "name": "Aspartimide side product",
                "mz": exact_mass_mh - 18.015,
                "cause": "Asp/Asn-rich motifs remain exposed to base-promoted cyclization during repeated deprotection.",
            }
        )
        mass_spectrum.append(
            {
                "mz": exact_mass_mh - 18.015,
                "abundance": max(4.0, 28.0 - (purity * 0.14)),
                "label": "[M-H2O+H]+",
            }
        )

    if summary.cys_count >= 2 and "Disulfide" not in params.topology:
        byproducts.append(
            {
                "name": "Cysteine oxidation dimer",
                "mz": (exact_mass_mh * 2.0) - 3.032,
                "cause": "Free cysteine residues can oxidize in air and form intermolecular disulfide-linked dimers.",
            }
        )

    if summary.hydrophobic_ratio > 0.48 and summary.length > 8:
        byproducts.append(
            {
                "name": "Aggregation-driven deletion series",
                "mz": exact_mass_mh - 113.084,
                "cause": "Long hydrophobic stretches favor resin aggregation and suppress diffusion of fresh activator.",
            }
        )

    if summary.difficult_ratio > 0.24 and summary.reagent_score < 0.9:
        optimizations.append("Upgrade coupling to HATU and add a double-coupling step for Arg/Cys/His-rich segments.")
    if summary.bulky_ratio > 0.28 and summary.temperature_score < 0.7:
        optimizations.append("Use microwave coupling at 75°C for sterically congested residues instead of room temperature.")
    if summary.hydrophobic_ratio > 0.42 and summary.solvent_score < 0.8:
        optimizations.append("Switch to NMP or DMF/DCM to reduce on-resin aggregation for hydrophobic stretches.")
    if summary.aspartimide_risk > 0.12:
        optimizations.append("Add Oxyma or HOBt during deprotection and avoid overly harsh basic exposure around Asp/Asn motifs.")
    if summary.cys_count >= 2 and "Disulfide" not in params.topology:
        optimizations.append("Protect free cysteine residues carefully and perform cleavage or workup under inert atmosphere.")
    if "Head-to-tail" in params.topology and summary.length < 6:
        optimizations.append("Head-to-tail cyclization is aggressive for short precursors; consider extending the linear precursor first.")

    if not optimizations:
        optimizations.append("Current condition is already near the tuned model optimum; keep the protocol and verify with analytical LC-MS.")
        optimizations.append("If scale-up is planned, preserve the same solvent and activator strength to avoid a purity drop.")

    mass_spectrum.sort(key=lambda peak: peak["mz"])

    return {
        "purity": purity,
        "yield": predicted_yield,
        "exactMass": exact_mass_mh,
        "massSpectrum": mass_spectrum,
        "byproducts": byproducts[:4],
        "optimizations": optimizations[:3],
        "training": training_info,
    }
