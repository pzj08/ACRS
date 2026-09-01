# 1. Experimental Setup

All analyses use the complete ACRS epoch-150 checkpoint `model_150.pt`
(SHA-256 `ca2314bedd9cdd7bf27acbcc6cbad7db4a632430851074002aebfbee066f5be4`).
No model was retrained. Evaluation uses the repository's full-utterance Vox1
shards, the original feature/CMVN path, cosine scoring without mean subtraction,
and WeSpeaker EER/minDCF (`p_target=0.01`). The diagnostic embeddings
match the archived full-system embeddings exactly for all 151,977 utterances.
The eight Correct score files also match all 1,316,888 archived score lines
exactly.

The evaluation set is the union of Only-CA5/10/15/20 and
Vox-CA5/10/15/20. Repository metadata provides continuous age for every
utterance in this union. The four CA lists overlap, so the gate-distance study
deduplicates them into 456,649 same-speaker positive pairs and uses true,
disjoint age-gap bins. The nominal `[0,5)` bin is empty because every source
list requires at least five years of mismatch. Pair-level confidence intervals
use 1,000 Poisson-bootstrap replicates.

Shuffle donors are deterministic, from a different speaker and a different
age group; all three shuffle conditions use the same donor for a target.
Near-Age selects the closest-age different-speaker donor, while Far-Age selects
the legal different-speaker age extreme. Interventions exchange
the exact donor-derived Stage-3 and/or Stage-4 residual gates while retaining
the target acoustic/shared/identity stream and target fusion condition. This
isolates residual suppression. Vox-O/E/H are not counterfactually evaluated
because 130 utterances outside the CA union lack continuous-age metadata; their
existing system results remain in `existing_results_summary.csv`.

# 2. Existing Ablation Findings

The primary comparison uses one uniform protocol without mean subtraction.
EER (%) on the controlled Only-CA conditions is:

| System | CA5 | CA10 | CA15 | CA20 |
|---|---:|---:|---:|---:|
| ResNet34 baseline | 2.061 | 3.306 | 5.159 | 7.232 |
| Full ACRS | 1.869 | 2.982 | 4.789 | 7.071 |
| no_age_loss | 1.893 | 3.284 | 5.554 | 7.974 |
| no_age_conditioning | 2.012 | 3.354 | 5.370 | 7.382 |
| random_gate_init | 2.175 | 3.618 | 6.177 | 8.987 |

Full ACRS improves over the controlled ResNet34 baseline on all four Only-CA
sets by 0.161--0.370 EER points and on Vox-CA by 0.429--1.366 points.
`no_age_loss` asks whether explicit age supervision is useful specifically as
age mismatch grows. Its absolute deficit relative to Full increases
monotonically from 0.024 EER points at CA5 to 0.302, 0.765, and 0.903 points at
CA10/15/20; relative reductions are 1.3%, 9.2%, 13.8%, and 11.3%. Vox-CA shows
the same monotonic absolute trend (0.147 to 1.186 points). This supports a
cross-age-specific benefit from age supervision rather than a uniform gain.

`no_age_conditioning` asks whether conditioning the suppression path is
necessary. Full is better on every CA set, by 0.143–0.581 points on Only-CA and
0.217–0.828 points on Vox-CA. The gap is not monotonic through CA20, but its
consistent direction supports the conditioning design. `random_gate_init` is
uniformly weaker and is best treated as an optimization/design diagnostic, not
as the main scientific comparison. No dataset-specific scoring protocol was
selected.

# 3. Gate Specialization

Different age conditions produce materially different channel patterns. At
Stage 3, age-group gate means occupy a narrow 0.385–0.394 range, yet the maximum
pairwise L1 distance between group centroids is 0.1135 (mean pairwise distance
0.0598). Stage 4 is stronger: overall means range from 0.404 to 0.429, while
maximum and mean centroid distances are 0.2090 and 0.0990. The Stage-4 heatmap
shows coherent, oppositely moving channel subsets rather than a uniform shift.

Thus, the data support the statement: **age conditioning primarily changes
which channels are suppressed rather than merely applying a global suppression
strength**, especially at Stage 4. Gate activation is not interpreted as the
quantity removed; actual suppression is analyzed separately. The age readout
also preserves meaningful ordering (true/predicted-age Spearman ρ=0.805,
MAE=5.13 years), although it underestimates the sparsely represented oldest
groups and should not be described as a calibrated age estimator.

# 4. Gate Distance vs Age Gap

| Stage | 5–10 years | 10–15 years | 15–20 years | ≥20 years | Spearman ρ (95% CI) | p-value |
|---|---:|---:|---:|---:|---:|---:|
| Stage 3 | 0.07198 | 0.06927 | 0.07218 | 0.07055 | 0.0100 [0.0071, 0.0128] | 1.21×10⁻¹¹ |
| Stage 4 | 0.05508 | 0.06014 | 0.06285 | 0.07424 | 0.1484 [0.1458, 0.1512] | <10⁻³⁰⁰ |

Stage 4 is approximately monotonic: the mean per-channel change grows in every
successive true-gap bin, and cosine distance gives the same conclusion
(ρ=0.1359). Stage 3 is non-monotonic and its statistically nonzero correlation
is scientifically negligible given the very large pair count; cosine distance
is equally weak (ρ=0.0106). Therefore, “suppression patterns diverge with age
gap” is supported for Stage 4, not as a blanket statement about both stages.
The bootstrap treats unique pairs as observations; repeated utterances imply
that its intervals should be read descriptively rather than as speaker-cluster
inference.

# 5. Actual Suppression Magnitude

The actual removed residual is measurable without changing the model:
`||gate ⊙ residual_candidate||F / ||identity_input||F`. Stage-4 mean suppression
ratio increases monotonically across age groups from 0.608 (`<21`) to 0.671
(`≥71`). Stage 3 increases from 0.553 for ages 21–30 to 0.594 for 61–70, followed
by 0.589 in the small `≥71` group (N=148).

For same-speaker pairs, the mean Stage-4 suppression ratio rises from 0.6246 at
5–10 years to 0.6484 beyond 20 years, with ρ=0.3260
[0.3232, 0.3286]. Its within-pair absolute difference also grows with gap
(ρ=0.1398). Stage 3 has a weaker relation for pair means (ρ=0.1134) and no
meaningful relation for within-pair differences (ρ=−0.0044). The amount
actually subtracted therefore varies with age and age mismatch primarily at
Stage 4.

# 6. Counterfactual Age Intervention

EER (%) under cosine scoring without mean subtraction is:

| Dataset | Correct | Shuffle-S3 | Shuffle-S4 | Shuffle-Both | Far-Age |
|---|---:|---:|---:|---:|---:|
| Only-CA5 | 1.869 | 3.683 | 2.909 | 6.429 | 9.005 |
| Only-CA10 | 2.982 | 4.700 | 4.483 | 7.910 | 14.063 |
| Only-CA15 | 4.789 | 6.684 | 6.187 | 9.843 | 17.362 |
| Only-CA20 | 7.071 | 8.726 | 7.904 | 11.545 | 17.703 |
| Vox-CA5 | 3.359 | 6.718 | 5.128 | 10.644 | 16.195 |
| Vox-CA10 | 4.676 | 7.659 | 6.822 | 12.100 | 21.882 |
| Vox-CA15 | 7.211 | 10.094 | 9.350 | 14.404 | 25.875 |
| Vox-CA20 | 9.604 | 12.071 | 11.012 | 16.095 | 25.042 |

Incorrect conditioning degrades EER on every dataset and every intervention.
Across Only-CA, Shuffle-S3 adds 1.65–1.89 points, Shuffle-S4 adds 0.83–1.50,
Shuffle-Both adds 4.47–5.05, and Far-Age adds 7.14–12.57. Across Vox-CA, the
corresponding ranges are 2.47–3.36, 1.41–2.15, 6.49–7.42, and 12.84–18.66.
All intervention minDCFs are also worse than Correct. Far-Age has the highest
EER in every case, although its minDCF is not always above Shuffle-Both.

The size of the EER degradation does **not** increase monotonically from CA5
to CA20: Shuffle-Both is nearly flat through CA15 and smaller at CA20, while
Far-Age peaks at CA15 before declining. The intervention strongly establishes
the need for the correct condition, but it does not establish that
intervention sensitivity grows continuously with benchmark severity.

## Near-Age versus Far-Age

Near-Age uses the different-speaker donor with the closest continuous age
(mean gap 0.00174 years; median 0.00055), while Far-Age uses the most distant
legal donor (mean gap 46.13 years). All other extraction and scoring choices
are identical. The comparison is:

| Dataset | Correct EER/minDCF | Near-Age EER/minDCF | Far-Age EER/minDCF | Near→Far ΔEER | Near→Far ΔminDCF |
|---|---:|---:|---:|---:|---:|
| Only-CA5 | 1.869 / 0.167 | 4.816 / 0.454 | 9.005 / 0.480 | +4.190 | +0.026 |
| Only-CA10 | 2.982 / 0.255 | 7.023 / 0.526 | 14.063 / 0.787 | +7.040 | +0.261 |
| Only-CA15 | 4.789 / 0.331 | 10.287 / 0.643 | 17.362 / 0.927 | +7.075 | +0.284 |
| Only-CA20 | 7.071 / 0.410 | 13.200 / 0.754 | 17.703 / 0.995 | +4.504 | +0.241 |
| Vox-CA5 | 3.359 / 0.286 | 8.157 / 0.604 | 16.195 / 0.636 | +8.038 | +0.032 |
| Vox-CA10 | 4.676 / 0.349 | 10.244 / 0.641 | 21.882 / 0.821 | +11.638 | +0.180 |
| Vox-CA15 | 7.211 / 0.470 | 14.049 / 0.780 | 25.875 / 0.956 | +11.826 | +0.177 |
| Vox-CA20 | 9.604 / 0.601 | 17.683 / 0.891 | 25.042 / 0.995 | +7.359 | +0.105 |

Far-Age is consistently worse than Near-Age: 8/8 sets by EER and 8/8 by
minDCF. This directly supports an age-distance effect in the residual
condition. However, Near-Age itself remains substantially worse than Correct,
despite almost identical donor age. The learned donor gate therefore contains
utterance-dependent variation beyond scalar continuous age, or requires
target-specific alignment; Near-Age should not be treated as a surrogate for
the correct target condition. Near→Far degradation is also non-monotonic over
CA5/10/15/20, so the result supports consistent near/far ordering rather than
a linear severity law.

Sanity checks passed. Self-intervention and repeated fixed-donor inference have
zero max/mean absolute difference; minimum self cosine is 0.99999982. All six
embedding-norm means equal 1.0, with the same numerical range
(0.99999982–1.00000012). The changes are therefore not a norm collapse.

# 7. Stage-wise Role

Stage 3 is **performance-dominant under intervention**: its shuffle causes a
larger EER increase than Stage 4 on all eight datasets. Stage 4 is
**mechanistically more age-ordered**: its gate distance and actual suppression
magnitude track true age gap much more strongly. Joint shuffling is worse than
either single-stage intervention everywhere, so the stages are complementary,
but they play different observable roles. “Stage3 dominant” is appropriate
for verification sensitivity; it would be inaccurate for age-gap
specialization.

# 8. Mechanistic Evidence Chain

1. **Age supervision → meaningful age representation: SUPPORTED.** The age
   readout has ρ=0.805 with true age, and the no-age-loss deficit grows with
   controlled age mismatch. Calibration at old ages remains limited.
2. **Age representation → age-dependent gate pattern: PARTIALLY SUPPORTED.**
   Group centroids differ strongly at both stages, but only Stage 4 changes
   coherently with within-speaker age gap.
3. **Age-dependent gate → selective identity refinement: SUPPORTED.** The
   implementation subtracts gated residuals, channel patterns specialize by
   age, and actual suppression ratios vary systematically, especially at
   Stage 4.
4. **Correct age conditioning → better cross-age verification: SUPPORTED.**
   Every wrong-condition intervention increases EER, while all numerical
   sanity checks pass.
5. **Stronger mismatch → larger need for age-conditioned refinement:
   PARTIALLY SUPPORTED.** The age-loss ablation gap and Stage-4 gate/suppression
   analyses support it, and Far-Age is worse than Near-Age on every set.
   However, Near→Far degradation is non-monotonic over benchmark severity and
   Stage 3 gate distance is essentially flat.

Overall, the experiments support the paper's calibrated core claim: ACRS uses
age as a conditioning signal to select residual components whose removal
matters for verification. They do not support a stronger claim that every
stage responds monotonically to age gap or that age is globally erased.

# 9. Strongest Finding

**The strongest experimental finding is: Far-Age is worse than Near-Age on all
eight sets by EER and minDCF, adding 4.19–11.83 EER points with target identity
and scoring unchanged.**

# 10. Second Strongest Finding

**Stage-4 gate distance rises approximately monotonically from 0.0551 at 5–10
years to 0.0742 beyond 20 years (ρ=0.148), whereas Stage 3 shows no meaningful
age-gap trend (ρ=0.010).**

# 11. Claims We CAN Make

- ACRS learns age-dependent, channel-selective suppression patterns, most
  clearly at Stage 4.
- Stage-4 gate distance and actual suppression magnitude increase with true
  within-speaker age gap over the evaluated range.
- Replacing the target residual gates with mismatched donor gates degrades
  cross-age verification across every evaluated condition.
- Far-age donor gates are consistently worse than nearest-age donor gates by
  both EER and minDCF across all eight cross-age sets.
- Stage-3 conditioning is more important to verification performance, while
  Stage-4 conditioning is more systematically organized by age gap.
- Explicit age supervision yields an increasingly large Full-versus-ablation
  advantage as the controlled benchmark age mismatch grows.

# 12. Claims We CANNOT Make

- ACRS completely disentangles age and identity or fully removes age
  information.
- Gate activation itself measures the amount removed; the gated residual is
  the relevant quantity.
- Gate channels directly represent biological aging.
- ACRS proves a causal biological aging mechanism. The intervention is causal
  only with respect to the model's derived residual-gating condition.
- ACRS is universally better than global age invariance; no direct global
  invariance comparator was run here.
- All improvements are statistically significant; EER uncertainty was not
  estimated and pair reuse complicates naive independent-pair inference.
- Age information is unnecessary at inference because aging has been removed.
  The intervention demonstrates the opposite for this model.
- Both residual stages show monotonically increasing gate distance, or
  intervention degradation grows monotonically from CA5 to CA20.

# 13. ICASSP Recommendation

Use one two-panel mechanism figure in the main paper: the Stage-4 top-channel
age-group heatmap on the left and the Stage-3/4 true-gap gate-distance curve on
the right. This displays both selectivity and the stage-specific limitation.

Table 2 should contain two compact panels using EER only: (a) Full,
`no_age_loss`, and `no_age_conditioning` on Only-CA5/10/15/20; (b) Correct,
Near-Age, Shuffle-S3, Shuffle-S4, Shuffle-Both, and Far-Age on the same sets.
Put minDCF,
Vox-CA intervention rows, `random_gate_init`, cosine-distance
robustness, full suppression tables, donor mappings, and sanity details on
GitHub. The full per-pair CSV and age-prediction calibration are useful audit
artifacts but do not merit four-page body space.

The mechanism evidence is sufficient for the core insight if the paper uses
the calibrated wording “selective age-conditioned residual suppression.” It
is not sufficient for disentanglement, biological interpretation, universal
superiority, or two-stage monotonicity claims.
