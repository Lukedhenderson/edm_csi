import os
import sys
import json
import argparse
import subprocess


def resolve_path(path, config_dir):
    """Resolve a path relative to the config file's directory."""
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(config_dir, path))


# Keys that map directly to --key=value CLI flags on edm/train.py.
# Each entry: (json_key, cli_flag, is_flag_only)
#   is_flag_only=True means --flag with no value (for Click is_flag options)
OPTION_MAP = [
    ("outdir",    "--outdir",    False),
    ("data",      "--data",      False),
    ("cond",      "--cond",      False),
    ("arch",      "--arch",      False),
    ("precond",   "--precond",   False),
    ("duration",  "--duration",  False),
    ("batch",     "--batch",     False),
    ("batch_gpu", "--batch-gpu", False),
    ("cbase",     "--cbase",     False),
    ("cres",      "--cres",      False),
    ("lr",        "--lr",        False),
    ("ema",       "--ema",       False),
    ("dropout",   "--dropout",   False),
    ("augment",   "--augment",   False),
    ("xflip",     "--xflip",     False),
    ("fp16",      "--fp16",      False),
    ("ls",        "--ls",        False),
    ("bench",     "--bench",     False),
    ("cache",     "--cache",     False),
    ("workers",   "--workers",   False),
    ("desc",      "--desc",      False),
    ("nosubdir",  "--nosubdir",  True),
    ("tick",      "--tick",      False),
    ("snap",      "--snap",      False),
    ("dump",      "--dump",      False),
    ("seed",      "--seed",      False),
    ("transfer",  "--transfer",  False),
    ("resume",    "--resume",    False),
    ("dry_run",   "--dry-run",   True),
]

PATH_KEYS = {"outdir", "data", "transfer", "resume"}


def build_command(config, config_dir):
    edm_train = os.path.join(os.path.dirname(os.path.abspath(__file__)), "edm", "train.py")
    num_gpus = config.get("num_gpus", 1)

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={num_gpus}",
        edm_train,
    ]

    for json_key, cli_flag, is_flag_only in OPTION_MAP:
        value = config.get(json_key)
        if value is None:
            continue

        if is_flag_only:
            if value:
                cmd.append(cli_flag)
            continue

        if json_key in PATH_KEYS and isinstance(value, str) and value:
            value = resolve_path(value, config_dir)

        if isinstance(value, list):
            value = ",".join(str(v) for v in value)

        cmd.extend([cli_flag, str(value)])

    return cmd


def main():
    parser = argparse.ArgumentParser(description="Config-driven EDM training wrapper")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.json",
        help="Path to the JSON training configuration file",
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    config_dir = os.path.dirname(os.path.abspath(args.config))
    cmd = build_command(config, config_dir)

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
