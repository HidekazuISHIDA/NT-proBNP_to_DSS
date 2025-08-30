# STARD 2015 Mapping

This document maps each STARD item to where it is reported in the manuscript and how it is supported in this repository.

**Index test:** decision-tree probability with a pre-specified fixed decision rule (threshold = 0.50)  
**Reference standard:** N-terminal pro–B-type natriuretic peptide (NT-proBNP) > 300 pg/mL  
**Setting/design:** single-center, retrospective; consecutive encounters  
**Missing data:** complete-case only; no imputation  
**Uncertain results:** none for index test; invalid NT-proBNP re-measured and excluded if unresolved

| No. | Item (abridged) | Where reported | Notes |
|---:|---|---|---|
| 1 | Identify as diagnostic accuracy study | Abstract (Methods/Results) | AUROC; sensitivity/specificity with CIs |
| 2 | Structured abstract | Abstract | Background, Methods, Results, Conclusions |
| 3–4 | Background and intended use | Introduction | Screening/triage; DSS integration |
| 5 | Study design | Methods – Study Design | Retrospective, single center |
| 6 | Eligibility criteria | Methods – Study Design / Outcome / Predictors | Same-encounter NT-proBNP required; complete-case |
| 7–9 | Setting, participants, time frame | Methods – Study Design | Consecutive encounters; derivation and temporal external windows |
| 10a | Index test details | Methods – Predictors / Model / Statistical Analysis | Standardization; LASSO feature selection; decision tree; fixed rule |
| 10b | Reference standard details | Methods – Laboratory / Outcome | Routine assay; >300 pg/mL threshold |
| 11 | Rationale for reference standard | Introduction / Outcome | Clinically relevant screening threshold |
| 12a | Index test positivity cutoff | Methods – Model / Middleware / Statistical Analysis | **Pre-specified fixed 0.50; no post-hoc tuning** |
| 12b | Reference standard cutoff | Methods – Outcome | **>300 pg/mL** pre-specified |
| 13a | Blinding (index vs reference) | Methods – Indeterminate/invalid handling | Model uses labs only; reference not used by model |
| 13b | Blinding (reference vs index) | Methods – Laboratory Setting | Routine measurement independent of model |
| 14 | Statistical methods | Methods – Statistical Analysis | **DeLong** for AUROC; **Wilson** for other metrics |
| 15 | Handling of indeterminate results | Methods – Indeterminate/invalid handling | NT-proBNP re-measured or excluded; no equivocal index category |
| 16 | Handling of missing data | Methods – Statistical / Predictors | Complete-case; no imputation |
| 17 | Analyses of variability | Methods – Middleware (Subgroups) / Results | Age, sex, eGFR; bootstrap CIs |
| 19 | Participant flow | Figure S1 | Train/test split; DSS deployment; temporal external set |
| 20 | Baseline characteristics | Table 1 | Abbreviations spelled out |
| 23 | Cross-tabulation | Figures 4–5 | Confusion matrices (fixed rule) |
| 24 | Estimates with precision | Table 2 | AUROC (DeLong) and other metrics (Wilson) with 95% CIs |
| 26–27 | Limitations and implications | Discussion | Spectrum/verification bias; DSS deployment implications |

**Reproducibility pointers**

- `scripts/make_table2.py` — reproduces Table 2 with DeLong/Wilson CIs at threshold 0.50.  
- `notebooks/` — optional notebooks to regenerate figures using the synthetic dataset.  
- See **README → Evaluation** for exact commands.
