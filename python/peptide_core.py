from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PeptideSynthesisParams:
    sequence: str
    topology: str
    coupling_reagent: str
    solvent: str
    temperature: str
    cleavage_time: str


@dataclass(frozen=True)
class FeatureSummary:
    length: int
    length_norm: float
    bulky_ratio: float
    difficult_ratio: float
    hydrophobic_ratio: float
    charged_ratio: float
    breaker_ratio: float
    sulfur_ratio: float
    aspartimide_risk: float
    reagent_score: float
    topology_complexity: float
    temperature_score: float
    solvent_score: float
    cleavage_score: float
    sequence_complexity: float
    longest_hydrophobic_run: float
    cys_cyclization_fit: float
    cys_count: int
    sensitive_ratio: float


@dataclass(frozen=True)
class FeatureBundle:
    residues: List[str]
    feature_names: List[str]
    vector: List[float]
    summary: FeatureSummary


AA_MASS: Dict[str, float] = {
    "Gly": 57.02146,
    "Ala": 71.03711,
    "Ser": 87.03203,
    "Pro": 97.05276,
    "Val": 99.06841,
    "Thr": 101.04768,
    "Cys": 103.00919,
    "Ile": 113.08406,
    "Leu": 113.08406,
    "Asn": 114.04293,
    "Asp": 115.02694,
    "Gln": 128.05858,
    "Lys": 128.09496,
    "Glu": 129.04259,
    "Met": 131.04049,
    "His": 137.05891,
    "Phe": 147.06841,
    "Arg": 156.10111,
    "Tyr": 163.06333,
    "Trp": 186.07931,
}

AMINO_ACIDS = list(AA_MASS.keys())

FEATURE_NAMES = [
    "length_norm",
    "bulky_ratio",
    "difficult_ratio",
    "hydrophobic_ratio",
    "charged_ratio",
    "breaker_ratio",
    "sulfur_ratio",
    "aspartimide_risk",
    "reagent_score",
    "topology_complexity",
    "temperature_score",
    "solvent_score",
    "cleavage_score",
    "sequence_complexity",
    "longest_hydrophobic_run",
    "cys_cyclization_fit",
]

TERMINAL_GROUPS = {"H", "OH", "NH2"}
BULKY_RESIDUES = {"Val", "Ile", "Thr", "Pro", "Phe", "Trp", "Tyr", "Leu"}
DIFFICULT_RESIDUES = {"Arg", "Cys", "His", "Asn", "Gln"}
HYDROPHOBIC_RESIDUES = {"Ala", "Val", "Ile", "Leu", "Phe", "Trp", "Tyr", "Met", "Pro"}
CHARGED_RESIDUES = {"Asp", "Glu", "Lys", "Arg", "His"}
BREAKER_RESIDUES = {"Gly", "Pro"}
SULFUR_RESIDUES = {"Cys", "Met"}
ASPARTIMIDE_RESIDUES = {"Asp", "Asn"}
SENSITIVE_RESIDUES = {"Asp", "Asn", "Cys", "Met"}


def _ratio(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def _count_residues(residues: List[str], target: set[str]) -> int:
    return sum(1 for residue in residues if residue in target)


def _longest_run(residues: List[str], target: set[str]) -> int:
    best = 0
    current = 0
    for residue in residues:
        if residue in target:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def parse_peptide_sequence(sequence: str) -> List[str]:
    return [
        part.strip()
        for part in sequence.split("-")
        if part.strip() and part.strip() not in TERMINAL_GROUPS
    ]


def reagent_score(reagent: str) -> float:
    if "HATU" in reagent:
        return 1.0
    if "PyBOP" in reagent:
        return 0.84
    if "HBTU" in reagent:
        return 0.66
    return 0.5


def topology_complexity(topology: str) -> float:
    if "Head-to-tail" in topology:
        return 0.95
    if "Disulfide" in topology:
        return 0.76
    return 0.12


def temperature_score(temperature: str) -> float:
    if "90" in temperature:
        return 1.0
    if "75" in temperature:
        return 0.72
    return 0.25


def solvent_score(solvent: str) -> float:
    if "NMP" in solvent:
        return 0.86
    if "DMF/DCM" in solvent:
        return 0.74
    return 0.55


def cleavage_score(cleavage_time: str) -> float:
    if "4" in cleavage_time:
        return 1.0
    if "3" in cleavage_time:
        return 0.72
    return 0.44


def extract_feature_bundle(params: PeptideSynthesisParams) -> FeatureBundle:
    residues = parse_peptide_sequence(params.sequence)
    length = max(1, len(residues))

    bulky_ratio = _ratio(_count_residues(residues, BULKY_RESIDUES), length)
    difficult_ratio = _ratio(_count_residues(residues, DIFFICULT_RESIDUES), length)
    hydrophobic_ratio = _ratio(_count_residues(residues, HYDROPHOBIC_RESIDUES), length)
    charged_ratio = _ratio(_count_residues(residues, CHARGED_RESIDUES), length)
    breaker_ratio = _ratio(_count_residues(residues, BREAKER_RESIDUES), length)
    sulfur_ratio = _ratio(_count_residues(residues, SULFUR_RESIDUES), length)
    aspartimide_risk = _ratio(_count_residues(residues, ASPARTIMIDE_RESIDUES), length)
    cys_count = sum(1 for residue in residues if residue == "Cys")
    sequence_complexity = len(set(residues)) / length
    longest_hydrophobic_run = _longest_run(residues, HYDROPHOBIC_RESIDUES) / length
    reagent_strength = reagent_score(params.coupling_reagent)
    topology_score = topology_complexity(params.topology)
    thermal_score = temperature_score(params.temperature)
    solvent_strength = solvent_score(params.solvent)
    cleavage_strength = cleavage_score(params.cleavage_time)
    cys_cyclization_fit = min(1.0, cys_count / 4.0) if "Disulfide" in params.topology and cys_count >= 2 else 0.0
    sensitive_ratio = _ratio(_count_residues(residues, SENSITIVE_RESIDUES), length)
    length_norm = min(1.0, max(0.0, (length - 4) / 12))

    summary = FeatureSummary(
        length=length,
        length_norm=length_norm,
        bulky_ratio=bulky_ratio,
        difficult_ratio=difficult_ratio,
        hydrophobic_ratio=hydrophobic_ratio,
        charged_ratio=charged_ratio,
        breaker_ratio=breaker_ratio,
        sulfur_ratio=sulfur_ratio,
        aspartimide_risk=aspartimide_risk,
        reagent_score=reagent_strength,
        topology_complexity=topology_score,
        temperature_score=thermal_score,
        solvent_score=solvent_strength,
        cleavage_score=cleavage_strength,
        sequence_complexity=sequence_complexity,
        longest_hydrophobic_run=longest_hydrophobic_run,
        cys_cyclization_fit=cys_cyclization_fit,
        cys_count=cys_count,
        sensitive_ratio=sensitive_ratio,
    )

    vector = [
        summary.length_norm,
        summary.bulky_ratio,
        summary.difficult_ratio,
        summary.hydrophobic_ratio,
        summary.charged_ratio,
        summary.breaker_ratio,
        summary.sulfur_ratio,
        summary.aspartimide_risk,
        summary.reagent_score,
        summary.topology_complexity,
        summary.temperature_score,
        summary.solvent_score,
        summary.cleavage_score,
        summary.sequence_complexity,
        summary.longest_hydrophobic_run,
        summary.cys_cyclization_fit,
    ]

    return FeatureBundle(
        residues=residues,
        feature_names=FEATURE_NAMES,
        vector=vector,
        summary=summary,
    )


def calculate_exact_mass_mh(sequence: str, topology: str) -> float:
    residues = parse_peptide_sequence(sequence)
    exact_mass = 18.01528

    for residue in residues:
        exact_mass += AA_MASS.get(residue, 0.0)

    if "Head-to-tail" in topology:
        exact_mass -= 18.01528

    if "Disulfide" in topology and sum(1 for residue in residues if residue == "Cys") >= 2:
        exact_mass -= 2.016

    return exact_mass + 1.007276
