from __future__ import annotations

import argparse
import json

from peptide_core import PeptideSynthesisParams
from predictor import predict


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local peptide synthesis prediction with the Python model.")
    parser.add_argument("--model-family", default="annotated", choices=["annotated", "neural"])
    parser.add_argument("--data-source-hint", default="literature")
    parser.add_argument("--sequence", default="H-Gly-Ala-Val-Leu-Ile-OH")
    parser.add_argument("--topology", default="Linear")
    parser.add_argument("--coupling-reagent", default="HATU")
    parser.add_argument("--solvent", default="DMF")
    parser.add_argument("--temperature", default="Room Temperature")
    parser.add_argument("--cleavage-time", default="2 hours")
    args = parser.parse_args()

    result = predict(
        PeptideSynthesisParams(
            sequence=args.sequence,
            topology=args.topology,
            coupling_reagent=args.coupling_reagent,
            solvent=args.solvent,
            temperature=args.temperature,
            cleavage_time=args.cleavage_time,
        ),
        model_family=args.model_family,
        data_source_hint=args.data_source_hint,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
