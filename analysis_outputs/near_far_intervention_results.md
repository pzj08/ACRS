# Near-Age versus Far-Age intervention

All values use the existing mean-subtracted cosine protocol. EER is reported in percent.

| Dataset | Correct EER/minDCF | Near-Age EER/minDCF | Far-Age EER/minDCF | Near→Far ΔEER | Near→Far ΔminDCF |
|---|---:|---:|---:|---:|---:|
| only_ca5 | 1.848 / 0.164 | 4.760 / 0.445 | 9.211 / 0.490 | +4.451 | +0.045 |
| only_ca10 | 3.053 / 0.249 | 6.950 / 0.520 | 14.327 / 0.792 | +7.377 | +0.271 |
| only_ca15 | 4.990 / 0.327 | 10.033 / 0.634 | 17.506 / 0.932 | +7.473 | +0.298 |
| only_ca20 | 7.222 / 0.421 | 13.079 / 0.762 | 18.084 / 0.995 | +5.005 | +0.232 |
| vox_ca5 | 3.353 / 0.291 | 8.058 / 0.602 | 16.562 / 0.647 | +8.504 | +0.045 |
| vox_ca10 | 4.686 / 0.355 | 10.222 / 0.638 | 22.220 / 0.824 | +11.999 | +0.186 |
| vox_ca15 | 7.285 / 0.466 | 13.866 / 0.761 | 25.941 / 0.954 | +12.075 | +0.193 |
| vox_ca20 | 9.858 / 0.596 | 17.207 / 0.898 | 25.138 / 0.995 | +7.931 | +0.097 |

Far-Age is worse than Near-Age on 8/8 sets by EER and 8/8 sets by minDCF.
Near→Far increases EER by 4.451–12.075 points and minDCF by 0.045–0.298.
The ordering is fully consistent, although the degradation magnitude is not
monotonic from CA5 to CA20.
