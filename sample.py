import sys
import os
import json
import argparse
import pickle
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "edm"))
import dnnlib
from torch_utils import distributed as dist


def resolve_path(path, config_dir):
    """Resolve a path relative to the config file's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))


def load_net(path_net, device):
    with dnnlib.util.open_url(path_net, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)["ema"].to(device)
    return net


def build_class_labels(class_label, batch_size, net, device):
    """Build class label tensor from config value.

    None  -> unconditional (returns None)
    -1    -> random class per sample
    >= 0  -> fixed class index
    """
    if class_label is None:
        return None
    if net.label_dim == 0:
        return None
    if class_label == -1:
        return torch.eye(net.label_dim, device=device)[
            torch.randint(net.label_dim, size=[batch_size], device=device)
        ]
    idx = int(class_label)
    return torch.eye(net.label_dim, device=device)[
        idx * torch.ones(batch_size, dtype=torch.long, device=device)
    ]


def prior_sample(net, batch_size, M, N, num_steps, sigma_max, sigma_min, rho,
                 class_labels, device):
    """Standard EDM Euler ODE sampler (Algorithm 2 without stochasticity)."""
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    latents = torch.randn((batch_size, 2, M, N), device=device)
    x = latents.to(torch.float64) * t_steps[0]

    for t_cur, t_next in zip(t_steps[:-1], t_steps[1:]):
        denoised = net(x, t_cur, class_labels).to(torch.float64)
        score = (x - denoised) / t_cur
        x = x + (t_next - t_cur) * score

    samples_cplx = torch.view_as_complex(
        x.permute(0, -2, -1, 1).contiguous()
    )
    return samples_cplx.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Config-driven EDM prior sampling")
    parser.add_argument(
        "--config",
        type=str,
        default="sample_config.json",
        help="Path to the JSON sampling configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(args.config))

    path_net = config["path_net"]
    device = config.get("device", "cuda:0")
    M = config["M"]
    N = config["N"]
    num_samples = config.get("num_samples", 1)
    batch_size = config.get("batch_size", 1)
    num_steps = config.get("num_steps", 300)
    sigma_max = config.get("sigma_max", 5.0)
    sigma_min = config.get("sigma_min", 0.002)
    rho = config.get("rho", 7)
    seeds = config.get("seeds", [0])
    class_label = config.get("class_label", None)
    output_path = resolve_path(
        config.get("output_path", "prior_samples.npy"), config_dir
    )

    print("Loading network...")
    net = load_net(path_net, device)

    class_labels = build_class_labels(class_label, batch_size, net, device)

    all_samples = []
    for i, seed in enumerate(seeds):
        print(f"Seed {seed} ({i + 1}/{len(seeds)}), generating {num_samples} sample(s)...")
        torch.manual_seed(seed)
        for s in range(num_samples):
            samples = prior_sample(
                net, batch_size, M, N, num_steps,
                sigma_max, sigma_min, rho,
                class_labels, device,
            )
            all_samples.append(samples)

    result = np.concatenate(all_samples, axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.save(output_path, result)
    print(f"Saved {result.shape} to {output_path}")


if __name__ == "__main__":
    main()
