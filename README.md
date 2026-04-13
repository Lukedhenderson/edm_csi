# edm_csi

Config-driven EDM pipeline for MRI: **training**, **prior sampling**, and **reconstruction**. Clone the repo, edit a JSON config, and run. No code changes needed for standard experiments.

All relative paths in configs are resolved **relative to the config file's directory**, not the shell CWD.

## Setup

- Python environment with **PyTorch** (CUDA recommended).
- **`edm` submodule** at `edm/`. After cloning, run `git submodule update --init --recursive` and install its dependencies.
- For reconstruction: input data as **NumPy `.npy`** files.

## Entry points

| Task | Command | Config template |
|------|---------|-----------------|
| Training | `python train.py --config train_config.json` | [`train_config.example.json`](train_config.example.json) |
| Prior sampling | `python sample.py --config sample_config.json` | [`sample_config.example.json`](sample_config.example.json) |
| Reconstruction | `python main.py --config recon_config.json` | [`recon_config.example.json`](recon_config.example.json) |

---

## Training

Thin wrapper around `edm/train.py`. Reads JSON, builds the `torchrun` command, and runs it as a subprocess. All `edm/train.py` Click options are supported as JSON keys.

```bash
python train.py --config train_config.json
```

### Key config fields

| Key | Meaning |
|-----|--------|
| `num_gpus` | Number of GPUs (`--nproc_per_node`). Default `1`. |
| `outdir` | Output directory for checkpoints and logs. |
| `data` | Path to the training dataset (ZIP or directory). |
| `cond` | Class-conditional training (`true`/`false`). |
| `arch` | Network architecture: `ddpmpp`, `ncsnpp`, or `adm`. |
| `precond` | Preconditioning: `vp`, `ve`, or `edm`. |
| `duration` | Training duration in millions of images. |
| `batch` | Total batch size across all GPUs. |
| `lr`, `ema`, `dropout`, `augment` | Standard hyperparameters. |
| `transfer` | Path to a `.pkl` for transfer learning. |
| `resume` | Path to `training-state-*.pt` to resume. |

See [`train_config.json`](train_config.json) for the full set with defaults.

---

## Prior sampling

Generates unconditional or class-conditional samples from a trained EDM checkpoint and saves raw complex-valued output as `.npy`.

```bash
python sample.py --config sample_config.json
```

### Config fields

| Key | Meaning |
|-----|--------|
| `path_net` | EDM checkpoint (`.pkl` path or URL). |
| `device` | e.g. `cuda:0`. |
| `M`, `N` | Spatial resolution of generated samples. |
| `num_samples` | Independent samples per seed. |
| `batch_size` | Samples per forward pass. |
| `num_steps` | ODE solver steps. |
| `sigma_max`, `sigma_min`, `rho` | Noise schedule. |
| `seeds` | List of RNG seeds. |
| `class_label` | `null` = unconditional, `-1` = random, `>= 0` = fixed class. |
| `output_path` | Where to write the `.npy` result. |

---

## Reconstruction

Single-image diffusion posterior sampling (DPS). Loads k-space, coil maps, and an optional mask from `.npy` files.

```bash
python main.py --config recon_config.json
```

### Config fields

| Key | Meaning |
|-----|--------|
| `path_net` | EDM checkpoint (URL or local `.pkl`). |
| `device` | e.g. `cuda:0`. |
| `kspace_path` | Undersampled k-space, shape `(1, C, M, N)`. |
| `coils_path` | Coil sensitivity maps, broadcast-compatible with forward model. |
| `mask_path` | Optional sampling mask; omit or `null` for all-ones. |
| `output_path` | Where to write `recon.npy`. |
| `seeds` | RNG seeds (list). |
| `num_steps`, `sigma_max`, `sigma_min`, `rho`, `img_l_ss` | Sampler schedule and likelihood scale. |
| `class_label` | Conditioning label or `null`. |

### Data shapes

- **`kspace`**: 4-D `(1, C, M, N)`, complex or real dtype.
- **`coils`**: Must match the forward model in `recon_edm.py`.
- **`mask`**: Last two dims `(M, N)`, broadcast with `kspace`.
