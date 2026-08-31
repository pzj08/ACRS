# A. Gate Analysis paragraph

We observe age-dependent channel specialization in both residual gates despite
small changes in their global activation levels. Across age groups, the
Stage-3 gate mean varies only from 0.385 to 0.394, while the maximum L1 distance
between group centroids reaches 0.113. Stage 4 shows stronger organization,
with a maximum centroid distance of 0.209. For same-speaker pairs, Stage-4
mean per-channel gate distance increases approximately monotonically from
0.055 for 5–10-year gaps to 0.074 beyond 20 years (Spearman ρ=0.148), whereas
Stage 3 is non-monotonic (ρ=0.010). These results suggest that age conditioning
primarily selects which residual channels to suppress, with age-gap-ordered
specialization concentrated at Stage 4.

# B. Counterfactual Intervention paragraph

To test whether the learned age condition is functionally required, we retain
each target utterance's acoustic and identity streams while replacing only its
Stage-3/4 residual gates with different-speaker donors. Even the
nearest-age donor degrades every benchmark, indicating that matching scalar
age does not reproduce the target-specific condition. More importantly,
Far-Age is worse than Near-Age on all eight sets by both EER and minDCF.
Moving from Near to Far adds 4.45–7.47 EER points on Only-CA and 7.93–12.08
points on Vox-CA. This consistent ordering supports an age-distance effect in
the learned residual condition, although the degradation magnitude is not
monotonic from CA5 to CA20. These results support the functional role of
correct, target-specific age conditioning.

# C. Ablation paragraph

Explicit age supervision becomes more useful as the controlled age mismatch
increases. On Only-CA5/10/15/20, Full ACRS obtains 1.848%, 3.053%, 4.990%, and
7.222% EER, compared with 1.885%, 3.283%, 5.606%, and 8.255% without the age
loss. The absolute reduction therefore grows monotonically from 0.037 to
1.033 EER points; Vox-CA shows the same trend. Removing age conditioning also
degrades every cross-age result, yielding 2.007%, 3.412%, 5.540%, and 7.693%
on the four Only-CA sets. Together, these ablations suggest that explicit age
supervision supplies a useful conditioning representation and that using this
representation in the suppression path is necessary for the observed
cross-age gains.

# D. Three candidate key findings

1. Mismatched residual-suppression conditions degrade every evaluated
   cross-age benchmark, supporting the functional role of correct age
   conditioning rather than incidental use of the age stream.
2. Far-age residual conditions are consistently worse than nearest-age
   conditions across all eight cross-age sets by both EER and minDCF.
3. Stage-4 gate distance and actual suppression magnitude increase with true
   within-speaker age gap, while Stage 3 remains weakly age-gap ordered.

# E. Recommended Figure caption

**Age-conditioned residual specialization.** Left: age-group centroids for the
16 Stage-4 gate channels with the largest between-age variance. Right: mean
per-channel gate distance for unique same-speaker pairs in disjoint true
age-gap bins. Stage 4 changes approximately monotonically with age gap, whereas
Stage 3 is non-monotonic. Error bars denote 95% Poisson-bootstrap confidence
intervals.

# F. Recommended Ablation/Intervention Table caption

**Cross-age verification under training ablations and inference-only
conditioning interventions.** Removing age supervision or age-conditioned
suppression tests whether each learned component is necessary; replacing the
target residual gates with nearest- and farthest-age donors tests whether both
target specificity and donor age distance matter at inference. EER (%) is
reported under a single mean-subtracted cosine protocol; lower is better.
