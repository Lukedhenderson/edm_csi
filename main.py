import os
import json
import argparse
import numpy as np
import torch

from recon_edm import load_edm_model, run_posterior_sampling


def resolve_path(path, config_dir):
    """Resolve a path relative to the config file's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))


def load_array(path, name):
    """Load a .npy array with a clear error on failure."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{name}: file not found at {path}")
    arr = np.load(path, allow_pickle=False)
    print(f"  {name}: loaded {arr.shape} {arr.dtype} from {path}")
    return arr


def validate_inputs(kspace, coils, mask):
    """Check that array shapes are compatible before moving to GPU."""
    if kspace.ndim != 4:
        raise ValueError(
            f"kspace must be 4-D (1, C, M, N), got shape {kspace.shape}"
        )
    _, C, M, N = kspace.shape

    if coils.ndim < 2:
        raise ValueError(
            f"coils must be at least 2-D and broadcast with kspace, got shape {coils.shape}"
        )

    if mask is not None and mask.shape[-2:] != (M, N):
        raise ValueError(
            f"mask spatial dims {mask.shape[-2:]} do not match kspace spatial dims ({M}, {N})"
        )

    return M, N


def main():
    parser = argparse.ArgumentParser(description="EDM MRI Reconstruction")
    parser.add_argument(
        "--config",
        type=str,
        default="recon_config.json",
        help="Path to the JSON configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(args.config))

    # --- model / device ---
    path_net = config["path_net"]
    device = config.get("device", "cuda:0")

    # --- sampler hyperparameters ---
    seeds = config.get("seeds", [0])
    num_steps = config.get("num_steps", 300)
    sigma_max = config.get("sigma_max", 5.0)
    sigma_min = config.get("sigma_min", 0.002)
    rho = config.get("rho", 7)
    img_l_ss = config.get("img_l_ss", 1.0)
    class_label = config.get("class_label", None)

    # --- input / output paths ---
    kspace_path = resolve_path(config["kspace_path"], config_dir)
    coils_path = resolve_path(config["coils_path"], config_dir)
    mask_path = config.get("mask_path", None)
    output_path = resolve_path(config.get("output_path", "reconstruction.npy"), config_dir)

    # --- load input arrays ---
    print("Loading input data...")
    kspace = load_array(kspace_path, "kspace")
    coils = load_array(coils_path, "coils")

    mask = None
    if mask_path is not None:
        mask = load_array(resolve_path(mask_path, config_dir), "mask")

    M, N = validate_inputs(kspace, coils, mask)

    if mask is None:
        mask = np.ones_like(kspace)

    # --- move to GPU ---
    kspace_undersampled_gpu = torch.tensor(kspace * mask, device=device)
    mask_gpu = torch.tensor(mask, device=device)
    coils_gpu = torch.tensor(coils, device=device)

    # --- load EDM model ---
    print("Loading EDM model...")
    net, t_steps = load_edm_model(path_net, device, num_steps, sigma_max, sigma_min, rho)

    # --- reconstruct ---
    print("Running posterior sampling...")
    reconstruction = run_posterior_sampling(
        net=net,
        t_steps=t_steps,
        kspace_undersampled_gpu=kspace_undersampled_gpu,
        mask_gpu=mask_gpu,
        coils_gpu=coils_gpu,
        device=device,
        M=M,
        N=N,
        seeds=seeds,
        class_label=class_label,
        img_l_ss=img_l_ss,
    )

    # --- save ---
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, reconstruction)
    print(f"Saved reconstruction to {output_path}")


if __name__ == "__main__":
    main()
