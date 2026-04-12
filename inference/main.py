import os
import json
import argparse

from common import load_config_with_base
from inference.runners import run_edm_inference

def main():
    parser = argparse.ArgumentParser(description="EDM MRI Reconstruction Orchestrator")
    parser.add_argument('--config', type=str, default='configs/inference/recon_config.json', help='Path to the configuration file')
    parser.add_argument('--base-config', type=str, default='configs/base_config.json', help='Path to the base configuration file')
    args = parser.parse_args()

    # Load configuration with base config inheritance
    config = load_config_with_base(args.config, args.base_config)

    # Run inference
    results = run_edm_inference(config)
    print(f"Inference completed. Processed {len(results)} batches.")
