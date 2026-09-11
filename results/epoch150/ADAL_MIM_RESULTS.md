# ADAL and MIM evaluation results

Published on 2026-09-11. Both runs use seed 3409 and the single epoch-150
checkpoint (`model_150.pt`), full-utterance identity embeddings, and raw cosine
scoring without mean subtraction. Each file contains EER (%) and minDCF
(`Ptarget=0.01`, `Cmiss=Cfa=1`) for Vox1 O/E/H, Only-CA 5/10/15/20, and
Vox-CA 5/10/15/20. The result files are byte-for-byte copies of the original
`scores/baseline_cos_result` outputs, following this directory's existing format.

ADAL is the local method adaptation to the seed-3409 baseline controls.
MIM is the latest run completed on 2026-09-11, using the explicitly named
`eq9_logq_reduction: marginal_sum` local reproduction assumption: sum of
per-dimension conditional densities, rather than a joint density. It is not
an official author implementation. This is distinct from the older `eq9mean`
run completed on 2026-09-06. The new MIM supervisor records successful
exits for both training and evaluation.

## Source provenance

Timestamps below are the original result-file modification times (UTC+08:00).

### ADAL

- Result: [adal_seed3409/no_mean_subtraction.txt](adal_seed3409/no_mean_subtraction.txt)
- Experiment: `casv_adal_3409fair_seed3409_run2`
- Result timestamp: `2026-09-07T02:00:49+08:00`
- Source: `/work1/pzj/casv_adal_3409fair_seed3409_run2_eval_model150_nomsub_11trials/scores/baseline_cos_result`
- SHA-256: `e33ba862d8365001d0f7c4ce24283affbb7f12b08770d0f6e28ecf16085370ee`

### MIM

- Result: [mim_marginalsum_seed3409/no_mean_subtraction.txt](mim_marginalsum_seed3409/no_mean_subtraction.txt)
- Experiment: `casv_mim_acrs_fair_seed3409_marginalsum_20260909`
- Result timestamp: `2026-09-11T15:56:47+08:00`
- Source: `/work1/pzj/casv_mim_acrs_fair_seed3409_marginalsum_20260909_eval_model150_nomsub_11trials/scores/baseline_cos_result`
- SHA-256: `bdf522d2ab388c750ccf5a9c5d3ea0d6d8de32e2c8de18570956952a44b65ff9`
