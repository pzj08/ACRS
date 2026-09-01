# Training settings

The epoch-150 ACRS system was trained on the VoxCeleb2 development set with the Vox-CA training filter. The public recipe is [configs/acrs_resnet34_vox2.yaml](configs/acrs_resnet34_vox2.yaml). Data paths are repository-relative placeholders and should be changed to the local VoxCeleb, Vox-CA, MUSAN, and RIRS locations.

The input is a 200-frame crop of 80-bin log Mel filterbanks. Training uses four GPUs with 64 samples per GPU, giving a global batch size of 256. The probability of applying either MUSAN noise or RIRS reverberation is 0.6, speed perturbation is enabled, and SpecAugment is disabled.

Speaker classification uses ArcFace with scale 32 and a fixed margin of 0.2. The optimizer is SGD with momentum 0.9, Nesterov acceleration, and weight decay $10^{-4}$. The base learning rate is 0.0167. WeSpeaker's `ExponentialDecrease` schedule applies a six-epoch warm-up from zero and exponential decay over 150 epochs to $5\times10^{-5}$. Its distributed scale ratio is derived from the global batch size and equals 4 for this setup.

Continuous ages are converted to seven groups using boundaries 21, 31, 41, 51, 61, and 71 years. A value on a boundary belongs to the upper group. The age head uses normalized bin-center regression and four-bin classification. The effective training objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{spk}}+r(e)\,0.05\mathcal{L}_{\mathrm{age}},
$$

where $r(e)$ increases linearly to one over the first two epochs. The archived configuration records `lambda_consistency=0.02`, `lambda_smooth=0.0001`, and `lambda_path=0`. Counterfactual training is disabled, so the consistency and path terms are zero; the ACRS loss implementation does not add a smoothness term. These fields are retained in the YAML to match the saved epoch-150 experiment configuration.

Verification results in this repository use full-utterance embeddings and cosine scoring without mean subtraction.
