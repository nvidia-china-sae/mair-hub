# Speech Super-Resolution and Enhancement with Flow Matching

This repository provides a best-practice implementation of joint speech super-resolution and enhancement based on flow-matching.  
The model and code are adapted from [FLowHigh](https://arxiv.org/abs/2501.04926), originally designed for super-resolution only.

---

## Tutorial

The full step-by-step tutorial is available in the Jupyter Notebook:

```
Speech_Super_Resolution_and_Enhancement_with_Flow_Matching.ipynb
```

It covers:
- Dataset download and preparation
- Model modifications to support enhancement
- Configuration file explanation
- Training and inference workflow

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download datasets:

```bash
bash download_datasets.sh
```

2. Resample datasets:
```bash
bash resample_wavs.sh data/dns_challenge/noise/ data/dns_challenge/noise_24k/ 24000
bash resample_wavs.sh data/dns_challenge/rir/ data/dns_challenge/rir_24k/ 24000
```

3. Start training:

```bash
python train.py --config=./configs/config_libritts_r_noise_0.9_rir_0.5.json

# train a larger model for better quality
python train.py --config=./configs/config_libritts_r_noise_0.9_rir_0.5_12_layers.json
```

4. Run inference:

```bash
bash inference.sh
```

## Disclaimer

This best practice is intended for understanding and demonstrating the implementation principles of speech super-resolution and enhancement. In current version, the enhanced audio may contain artifacts. Its quality still needs to be further optimized for production use.
