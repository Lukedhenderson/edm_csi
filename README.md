# edm_csi

EDM-based MRI reconstruction (single slice/volume per run). **All runtime options live in a JSON config**—edit paths and hyperparameters there, then run `main.py`.

Everything `main.py` reads at run time comes from that JSON (plus `--config` to choose which file). You still need **`.npy` files on disk** (the config only points to them), a working **`edm/` checkout and its dependencies**, and a valid **`path_net`** checkpoint.

## Setup

- Python environment with **PyTorch** (CUDA recommended).
- **`edm` submodule** at `edm/` (used by `recon_edm.py` for `dnnlib` and the pickled network). Install whatever that submodule expects.
- Input data as **NumPy `.npy`** files (see below).

## Usage

```bash
python main.py --config recon_config.json
```

Paths in the config that are relative are resolved **relative to the config file’s directory**, not the shell’s current working directory.

Copy and fill [`recon_config.example.json`](recon_config.example.json) as a template, or edit [`recon_config.json`](recon_config.json).

## Config fields

| Key | Meaning |
|-----|--------|
| `path_net` | EDM checkpoint (URL or local `.pkl`). |
| `device` | e.g. `cuda:0`. |
| `kspace_path` | Undersampled k-space, shape `(1, C, M, N)`. |
| `coils_path` | Coil sensitivity maps, layout must broadcast with the forward model (same role as `coils` from the old dataloader). |
| `mask_path` | Optional sampling mask; omit or `null` for all-ones (full mask like the old TODO). |
| `output_path` | Where to write `recon.npy`. |
| `seeds` | RNG seeds (list; only the last seed’s result is kept today if you pass several). |
| `num_steps`, `sigma_max`, `sigma_min`, `rho`, `img_l_ss` | Sampler schedule and likelihood scale. |
| `class_label` | Conditioning label for the network, or `null` if unconditional. |

## Data

- **`kspace`**: 4-D `(1, C, M, N)`, complex or real dtype as supported by `torch.tensor`.
- **`coils`**: Must match how the forward in `recon_edm.py` expects to multiply and FFT (validate with your exported arrays).
- **`mask`**: If provided, last two dimensions must be `(M, N)` and broadcast with `kspace`.
