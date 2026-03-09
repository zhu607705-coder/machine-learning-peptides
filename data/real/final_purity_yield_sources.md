# Final Purity / Yield Source Scan

This file tracks public sources that are closer to the current UI target
(`final crude purity` / `isolated yield`) than the MIT `peptimizer` step-level
dataset.

## High-confidence primary sources

| Source | Link | What is available | Approx. usable rows | Fit for current UI |
| --- | --- | --- | ---: | --- |
| Peptimizer fast-flow peptide synthesis dataset | https://github.com/learningmatter-mit/peptimizer | 12,600 real experimental step records with `first_area`, `first_height`, `first_width`, `first_diff`; process variables and pre-chain sequence | 12,600 | Proxy only, not final purity/yield |
| Segetalins A-H, J and K (BJOC 2025) | https://www.beilstein-journals.org/bjoc/articles/21/202 | Table 1 provides 10 linear precursor crude-yield records with explicit sequences; final cyclic products reported in aggregate at >95% purity and practical isolated yields | 10 extracted (linear precursors) | Partial match |
| Resonant acoustic mixing enables solvent-less amide coupling in SPPS (Green Chem. 2025) | https://pubs.rsc.org/en/content/articlehtml/2025/gc/d5gc04067a | IKVAV: 95% crude purity, 81% isolated yield; Angiotensin (1-7): 94% crude purity, 70% isolated yield | 2 | Strong match but tiny |
| Improving Fmoc SPPS of Human Beta Defensin 3 (PMC 2022) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9603898/ | Table 1 includes condition-resolved yields for purified linear peptide and native folded peptide across resin/protection strategies | 12 extracted | Yield only / purity partial |
| Tetrahydropyranyl Backbone Protection for Enhanced Fmoc SPPS (PMC 2024) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12351424/ | Nigrocin-HLM crude purity improved 32% -> 54%; isolated yield improved 12% -> 24% | 2 | Strong match but tiny |
| Synthesis of Peptides Containing C-Terminal Methyl Esters Using Trityl Side-Chain Anchoring (PMC 2013) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3622458/ | One peptide with crude purity and isolated yield across two resin choices | 2 | Strong match but tiny |
| Continuous-Flow SPPS to Enable Rapid, Multigram Deliveries of Peptides (OPRD 2024) | https://pubs.acs.org/doi/10.1021/acs.oprd.4c00165 | Table 3 reports purity and isolated yield under different linear velocities; currently six thymopentin entries (1a-1f) are captured with snippet-level provenance and pending SI/full-text recheck | 6 extracted | Strong match but tiny |
| Parallel Peptide Purification with Catching Full-Length Sequence by Reversed-Phase Supports via a Traceless Linker System (PMC 2021) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8179278/ | Table 2 reports 20 sequence-resolved peptides with crude purity, final purity after PEC purification, and recovery | 20 | Strong fit for purified purity / recovery, especially purity-side expansion |
| High Throughput Synthesis of Peptide alpha-Thioesters (PMC 2008) | https://pmc.ncbi.nlm.nih.gov/articles/PMC3117248/ | Table with many peptide sequences and crude purity/yield values on a specific support | about 15 | Strong match, but chemistry/task narrower |
| Tea Bags for Fmoc Solid-Phase Peptide Synthesis (PMC 2021) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8399505/ | Table 2 reports 52 peptides with sequence, crude HPLC purity and purity-adjusted crude yield; 50 entries are mass-spec confirmed | 50 | Strong fit for crude purity / crude yield |
| Peptide Purification Transfer Study (PMC 2026) | https://pmc.ncbi.nlm.nih.gov/articles/PMC12890753/ | 23 synthetic peptides with defined impurity groups; average purified purity 92.9%, mean yield 32.0%; acetylated 95.5%/41.4%, non-acetylated 91.1%/25.3% | 23 | Strong match with sequence-specific purity data |
| ELISA Peptides (ResearchGate 2024) | https://www.researchgate.net/publication/294013748 | Table S3 provides 8 peptide sequences with HPLC purity percentages (90-98%) | 8 | Strong match with explicit purity percentages |
| HLA-E Peptide Production (Nature 2021) | https://www.nature.com/articles/s41598-021-96560-9 | Table 1 provides 13 HLA signal peptide sequences with production yields (mg/L) | 13 | Production yield data |
| Tumor Neoantigen Flow Synthesis (Nature 2019) | https://www.nature.com/articles/s41598-019-56943-5 | 29 IMPs (30-mer) and 48 ASPs (14-15-mer) with crude/purified purity and isolated yield data; 17 IMPs >=95% purity average 11% yield | 77 | Strong fit - multiple peptides with both purity and yield |
| Glucagon Impurity Study (Nature 2020) | https://www.nature.com/articles/s41598-020-61109-9 | Table 5 identifies 9 peptide impurity sequences from glucagon synthesis | 9 | Impurity characterization data |
| Tea Bags for Fmoc Solid-Phase Peptide Synthesis (PMC 2021) | https://pmc.ncbi.nlm.nih.gov/articles/PMC8399505/ | Table 2 reports 52 peptides with sequence, crude HPLC purity and purity-adjusted crude yield; 50 entries are mass-spec confirmed | 50 | Strong fit for crude purity / crude yield |
| Protocol for efficient solid-phase synthesis of peptides containing 1,2-HOPO (PMC 2020) | https://pmc.ncbi.nlm.nih.gov/articles/PMC7551356/ | Table 1 contains a few explicit purity/yield examples for HOPO-containing peptides | 4 | Partial match |
| Rapid flow-based synthesis of post-translationally modified peptides and proteins: a case study on MYC's transactivation domain (Sci. Adv./Chem. Sci. 2024) | https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc00481g | Several PTM-rich peptide examples with crude or purified purity plus isolated yield | about 5 | Strong match but tiny |
| Picking the tyrosine-lock: chemical synthesis of recifin A and analogues (Chem. Sci. 2024) | https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc01976h | Fragment and ligation yields for recifin A synthesis; purity less explicit | about 4 | Yield only / sequence often missing |
| Impact of N-Terminal PEGylation on Synthesis and Purification of Peptide-Based Cancer Epitopes for PDAC (PMC 2024) | https://pmc.ncbi.nlm.nih.gov/articles/PMC11325526/ | Table 3 reports 12 crude purity/yield rows and 12 purified purity/recovery rows across four sequence families | 24 | Strong fit for mixed-stage purity/yield |
| A Simple and Flexible Synthesis of Sulfotyrosine-Containing Peptides (PSC 2016) | https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/psc.2868 | Table 4 reports six sequence-resolved sulfotyrosine peptides with isolated yield after HPLC and final purity | 6 | Strong fit for sequence + isolated yield, but modified chemistry |
| Development of a straightforward synthesis route to hydrocarbon stapled BCL9 alpha-helical peptides with improved Wnt/β-catenin inhibitory activity (RSC Chem. Biol. 2025) | https://pubs.rsc.org/en/content/articlelanding/2025/cb/d4cb00368d | Table 1 reports four sequence-resolved stapled peptides with isolated yield and final purity after RP-HPLC | 4 | Strong fit for sequence + isolated yield, but stapled peptides |
| Accelerated Multiphosphorylated Peptide Synthesis (AMPS) (PMC 2022) | https://pmc.ncbi.nlm.nih.gov/articles/PMC9397535/ | Table 2 reports 8 multiphosphorylated peptide sequences with crude HPLC purity (7.7-37.2%) and isolated yields (3.6-33.6%) using AMPS method with elevated temperature Fmoc deprotection | 8 | Strong fit for difficult sequence - multiple phosphorylation sites |
| Universal Peptide Synthesis via Solid-Phase Methods fused with Automation (Nature 2025) | https://www.nature.com/articles/s41467-025-62344-2 | Multiple difficult peptide sequences including ACP, 18A, GHRH, Semaglutide, NYAD-13 (stapled), Capitellacin (cyclic), N-Methyl-18A with crude/purified purity and isolated yield data | 7 | Strong fit - automated platform with various peptide modifications |

## Current conclusion

- The only large public dataset found so far is `peptimizer`, but it predicts
  step-level deprotection/aggregation proxies rather than final peptide purity
  or isolated yield.
- Public sources with true final purity/yield exist, but they are fragmented
  across papers and usually contain only 2 to 20 rows per paper.
- This means a robust supervised model for final purity/yield is still blocked
  by data volume and heterogeneity, not by model architecture alone.
- The manually curated literature dataset has now been expanded to
  `data/real/final_purity_yield_literature.csv` with 220 source-tracked rows
  across 20 sources.
- The workflow now benefits from target-specific ingestion: purity-only and
  yield-only literature rows can be retained for their own semantic heads
  instead of being discarded for missing the other target.

## Practical next step

If we want the UI target to stay as `purity + yield`, the most defensible route
is to build a manually curated literature dataset from these papers and their
supporting information, then train a small tabular model with explicit source
tracking.
