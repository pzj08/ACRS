# Near-Age versus Far-Age intervention

All values use cosine scoring without mean subtraction. EER is reported in percent.

| Dataset | Correct EER/minDCF | Near-Age EER/minDCF | Far-Age EER/minDCF | Near→Far ΔEER | Near→Far ΔminDCF |
|---|---:|---:|---:|---:|---:|
| only_ca5 | 1.869 / 0.167 | 4.816 / 0.454 | 9.005 / 0.480 | +4.190 | +0.026 |
| only_ca10 | 2.982 / 0.255 | 7.023 / 0.526 | 14.063 / 0.787 | +7.040 | +0.261 |
| only_ca15 | 4.789 / 0.331 | 10.287 / 0.643 | 17.362 / 0.927 | +7.075 | +0.284 |
| only_ca20 | 7.071 / 0.410 | 13.200 / 0.754 | 17.703 / 0.995 | +4.504 | +0.241 |
| vox_ca5 | 3.359 / 0.286 | 8.157 / 0.604 | 16.195 / 0.636 | +8.038 | +0.032 |
| vox_ca10 | 4.676 / 0.349 | 10.244 / 0.641 | 21.882 / 0.821 | +11.638 | +0.180 |
| vox_ca15 | 7.211 / 0.470 | 14.049 / 0.780 | 25.875 / 0.956 | +11.826 | +0.177 |
| vox_ca20 | 9.604 / 0.601 | 17.683 / 0.891 | 25.042 / 0.995 | +7.359 | +0.105 |

Far-Age is worse than Near-Age on 8/8 sets by EER and 8/8 sets by minDCF. Near→Far increases EER by 4.190–11.826 points and minDCF by 0.026–0.284. Consistent ordering: True.
