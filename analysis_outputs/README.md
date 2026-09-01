# ACRS inference-only mechanism analysis

## Scope and provenance

This directory contains inference-only analyses of the complete ACRS system.
No model was trained, no training configuration was changed, and the existing
embedding/scoring outputs were not overwritten. The evaluated checkpoint is
`/work1/pzj/casv_aqds_r021_seed3409/models/model_150.pt` (epoch 150; SHA-256
recorded in `cache/cache_verification.json`). The source configuration is
`/work1/pzj/casv_aqds_r021_seed3409/config.yaml`.

Evaluation reuses the recipe's VoxCeleb shard reader, full-utterance feature
extraction (`batch_size=1`, `whole_utt=True`, dither disabled), CMVN, cosine
scoring without mean subtraction, and WeSpeaker EER/minDCF definitions
(`p_target=0.01`, `c_miss=1`, `c_fa=1`). A direct-wav shortcut was rejected
because it did not reproduce the archived extraction exactly. The cache is
checked against the existing full-system `xvector.scp` before analysis.

## Confirmed ACRS mechanism

The shared ResNet stem ends after Stage 2 and feeds separate age and identity
streams. `residual3` operates on 128-channel Stage-3 identity features and
`residual4` on 256-channel Stage-4 identity features. At each stage, statistics
pooling of the corresponding age feature (channel means and standard
deviations over frequency and time) is passed through an MLP and sigmoid to
produce a `B × C` channel gate. The implemented update is exactly

```text
residual_candidate = Conv1x1(identity_before_suppression)
suppressed_residual = gate * residual_candidate
refined_identity = identity_before_suppression - suppressed_residual
```

Consequently, gate activation and removed residual magnitude are different
quantities. This analysis reports both. The suppression ratio is
`||gate * residual_candidate||_F / (||identity_before_suppression||_F + eps)`.

After Stage 4, the fusion block projects the age feature from 256 to 32
channels, concatenates it with refined identity features, predicts a spatial
sigmoid gate with a 1×1 convolution, and multiplies the identity tensor by that
gate before attentive statistics pooling.

The normal `forward()` already returns the Stage-3/4 residual gates, age
prediction/logits, and deep age/identity tensors; `return_layer_acts=True`
adds the suppression-block inputs and outputs. It also supports
`intervene_a3` and `intervene_a4`. These tensors replace the conditions passed
to Stage-3 residual suppression and to both Stage-4 residual suppression and
fusion, respectively. Default forward behavior is unchanged; `model.py` was
not modified for this analysis.

## Evaluation population and age metadata

The union of Only-CA5/10/15/20 and Vox-CA5/10/15/20 contains 1,316,888 trial
rows and 151,977 unique utterances. The positive same-speaker union contains
456,649 unique pairs. Trial lists reuse utterances extensively, so every
unique utterance is forwarded once per condition and then cached.

Continuous segment ages are available from the repository's actual Vox-CA
metadata (`/xmudata/pzj/vox-ca/vox1/segment2age.npy`) for all 151,977
utterances and all positive pairs. No ages were inferred from filenames. Age
groups follow the configured boundaries 21, 31, 41, 51, 61, and 71 years,
with an exact boundary assigned to the upper group. CA5/10/15/20 lists are
overlapping threshold-based conditions; gate-distance analysis therefore uses
their unique positive-pair union and mutually exclusive true-gap bins
`[0,5)`, `[5,10)`, `[10,15)`, `[15,20)`, and `[20,+inf)`.
The `[0,5)` bin has no pair because every source list requires at least a
five-year mismatch; it is retained as an explicit zero-count/NA row rather
than populated from unrelated trials.

Vox-O/E/H are retained in the existing-system summary but are not included in
the counterfactual table. Their utterance union extends beyond the Vox-CA
analysis population, and 130 of the additional utterances have no continuous
age entry in the repository metadata; a compliant Far-Age mapping therefore
cannot be constructed for the complete standard-trial population.

## Counterfactual intervention construction

The donor mapping is deterministic and every donor is a different speaker.
Shuffle-S3, Shuffle-S4, and Shuffle-Both use the same fixed donor from a
different configured age group. Near-Age uses the legal donor with the closest
continuous age, without an age-group constraint; Far-Age uses the legal donor
with the most distant continuous age among the available extrema.

Raw donor `a3/a4` maps cannot be passed directly between arbitrary
full-utterance examples because their time dimensions differ. The residual
blocks depend on those maps only through their channel gates, so Stage-3 and
Stage-4 interventions swap the exact cached donor gates. In accordance with
the experiment definition, the fusion block always retains the target's own
full-resolution `a4`; the intervention therefore isolates residual
suppression rather than mixing residual and fusion effects. It tests the
derived age-conditioning signal at each residual block, not an unmodified raw
activation tensor.

Near-Age and Far-Age both replace the Stage-3 and Stage-4 residual gates with
the corresponding donor gates. They differ only in donor age distance; target
features, target fusion, full-utterance extraction, and scoring are identical.

Self-intervention uses the model's native `intervene_a3/a4` arguments with the
utterance's own full-resolution activations. Determinism and embedding norms
are reported in `intervention_sanity_checks.md`.

## Reproducibility and generated files

`analysis/` contains the manifest, extraction, intervention, scoring, and
analysis scripts. All stochastic summaries use a fixed local random state.
Confidence intervals are 1,000-replicate Poisson bootstrap intervals; the
method is identified in the corresponding CSV. The final cache and generated
scientific reports are written only under `analysis_outputs/`.

`cache/acrs_diagnostics.pt` stores utterance/speaker/age fields, the normal
speaker embedding, both residual gates, age regression and classification
outputs, pooled Stage-3/4 age representations, the temporally pooled low-rank
fusion condition, and both exact suppression ratios. Full variable-length
`a3/a4` maps are not duplicated in the cache because that would require a
multi-gigabyte variable-shape activation store; the maps are retained only
during each forward pass when exact residual candidates and self-intervention
checks are computed.

The public repository reports all verification results without mean
subtraction. It contains the analysis scripts, aggregate CSV results, figures,
reports, and compact verification records. Large utterance-level
caches, the raw positive-pair table, the complete donor mapping, and the raw
pair-level gate-distance table are reproducible generated artifacts and are
excluded from version control.

The extraction and analysis status, including any missing output or failed
sanity check, is updated in this file when the run completes.

## Completion status

All Priority 1–9 analyses are complete for the eight CA evaluation sets. The
normal cache reproduces 151,977 archived embeddings exactly, and Correct
scoring reproduces all 1,316,888 archived score lines exactly. Self
intervention and repeated fixed-donor intervention have zero max/mean absolute
embedding difference; all condition norms remain at one.

The main empirical result is stage-specific. Stage-4 gate distance increases
approximately monotonically with true age gap (ρ=0.1484), while Stage 3 is
non-monotonic and effectively flat (ρ=0.0100). Actual Stage-4 suppression has a
stronger age-gap relation (pair-mean ρ=0.3260). Wrong residual conditions raise
EER in every test: joint-stage shuffling adds 4.47–7.42 EER points and Far-Age
adds 7.14–18.66 points. Stage-3 shuffling is more damaging to verification than
Stage-4 shuffling, although Stage 4 is more age-ordered. Intervention
degradation is not monotonic from CA5 to CA20.

The Near-Age extension uses different-speaker donors with a mean continuous
age gap of 0.00174 years, versus 46.13 years for Far-Age. Far-Age is worse than
Near-Age on all eight CA sets by both EER and minDCF. Near→Far adds
4.190–11.826 EER points and 0.026–0.284 minDCF. Near-Age is still worse than
Correct on every set, so donor gates retain target/utterance-dependent
variation beyond scalar continuous age.

No requested CA analysis was skipped. The only omitted optional experiment is
counterfactual Vox-O/E/H evaluation, for the metadata reason documented above.
No training or model source file was modified.
