# Literature Dataset Notes

This dataset is a manually curated, source-tracked table of public peptide
synthesis examples that report explicit final purity and/or yield values close
to the current app target.

Files:

- `data/real/final_purity_yield_literature.csv`
- `data/real/final_purity_yield_sources.md`

## Schema

- `record_id`: stable local row identifier
- `source_id`: stable source identifier
- `source_title`: paper title
- `source_url`: primary-source URL
- `publication_year`: publication year
- `peptide_label`: peptide name or paper entry label
- `sequence`: peptide sequence when explicitly available
- `topology`: `linear` when explicit; blank when not recoverable quickly
- `purity_pct`: reported purity value
- `purity_stage`: what that purity refers to
- `yield_pct`: reported yield value
- `yield_stage`: what that yield refers to
- `yield_basis`: author-reported basis for yield
- `condition_summary`: compressed experimental context
- `source_note`: exact table or text location used during manual extraction

## Important caveats

- This is not a homogeneous benchmark. The papers mix different chemistries,
  supports, protecting-group strategies, and purification conventions.
- `purity_pct` is not always the same concept across papers:
  some rows are crude HPLC purity, others are final product purity.
- `yield_pct` is not always the same concept either:
  some rows are crude recovery relative to resin substitution, others are
  isolated yield after purification.
- Missing fields are intentional. If a sequence or topology could not be
  extracted quickly and confidently from the primary source, the field is left
  blank instead of inferred.

## Recommended use

- Use this file as a seed dataset for careful literature expansion.
- Before model training, filter rows by compatible label semantics.
- For example, one conservative subset is:
  `purity_stage == final_product` and `yield_stage == isolated`
- Another is:
  `purity_stage == crude_hplc` and `yield_stage == crude`

## Current source coverage

- Amyloid beta hydrophobic peptide synthesis paper: 14 rows
- Peptide alpha-thioester support paper: 17 rows
- PDAC PEGylation peptide synthesis paper: 24 rows
- Tea Bags for Fmoc SPPS paper: 50 rows
- HOPO-containing peptide protocol paper: 4 rows
- MYC AFPS PTM peptide paper: 5 rows
- Recifin A chemical synthesis paper: 4 rows
- Resonant acoustic mixing SPPS paper: 3 rows
- THP backbone protection paper: 2 rows
- Trityl side-chain anchoring paper: 2 rows
- Sulfotyrosine-containing peptide synthesis paper: 6 rows
- Hydrocarbon stapled BCL9 peptide synthesis paper: 4 rows
- Segetalins A-H, J, K synthesis paper (BJOC 2025): 10 rows
- Human Beta Defensin 3 Fmoc SPPS optimization paper: 12 rows
- Continuous-flow SPPS OPRD 2024 paper: 6 rows

Current total: 163 source-tracked rows across 15 primary sources.

Notes:

- The `Tea Bags for Fmoc Solid-Phase Peptide Synthesis` source contributes
  50 sequence-resolved rows from Table 2, excluding two entries whose mass
  spectrometry analysis did not confirm the expected peptide.
- The `PDAC` paper contributes both crude rows and purified-recovery rows for
  the same sequence families, which improves stage coverage but also increases
  within-source correlation.
- The newly added `sulfotyrosine` and `stapled BCL9` sources improve coverage
  of sequence-resolved `isolated yield` rows, but they also introduce more
  chemically specialized peptides. Downstream analysis should therefore include
  study-level stratification and sensitivity checks rather than treating these
  rows as fully homogeneous with simple linear SPPS products.
- The newest `sulfotyrosine` and `stapled BCL9` rows were captured from direct
  table summaries surfaced during source search. They are suitable for
  exploratory modeling and sensitivity analysis, but if the dataset is later
  used for publication-grade benchmarking, those rows should be rechecked
  against the publisher HTML/PDF tables line by line.
- The newly added `segetalins_bjoc_2025` rows currently represent Table 1
  linear precursor records with explicit crude-yield values. The same paper
  reports final cyclic products at >95% purity and practical isolated yields
  in aggregate, but per-target cyclic yields are not explicitly tabulated in
  machine-readable form in the main text, so they were not inferred here.
- The `hbd3_fmoc_2022` rows are taken directly from Table 1 yield entries and
  are explicitly split by product stage (`purified linear peptide` vs
  `native peptide`) and synthesis/folding condition.
- The `continuous_flow_oprd_2024` rows now include all six thymopentin
  velocity variants listed in Table 3 (`1a`-`1f`). Because ACS full-text
  access remains partially restricted in automated fetch, these values are
  still tagged as snippet-derived and should be rechecked line-by-line against
  SI/PDF or institutional full-text export before publication-grade use.
