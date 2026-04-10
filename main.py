import os
import json
import argparse
import numpy as np
import torch

from fastMRI_dataloader import loader # user's defined loader
from recon_edm import load_edm_model, run_posterior_sampling

def main():
    parser = argparse.ArgumentParser(description="EDM MRI Reconstruction Orchestrator")
    parser.add_argument('--config', type=str, default='recon_config.json', help='Path to the configuration file')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # 1. Setup Config
    path_net = config.get('path_net', '')
    device = config.get('device', 'cuda:0')
    save_dir = config.get('save_dir', 'reconstructions')
    
    seeds = config.get('seeds', [0])
    num_steps = config.get('num_steps', 300)
    sigma_max = config.get('sigma_max', 5.0)
    sigma_min = config.get('sigma_min', 0.002)
    rho = config.get('rho', 7)
    img_l_ss = config.get('img_l_ss', 1.0)
    class_label = config.get('class_label', None)
    
    M = config.get('M', None)
    N = config.get('N', None)

    # 2. Load Model
    print("Loading EDM Model...")
    net, t_steps = load_edm_model(path_net, device, num_steps, sigma_max, sigma_min, rho)

    os.makedirs(save_dir, exist_ok=True)

    # 3. Execution Loop
    print("Starting Reconstruction Loop...")
    for batch_idx, (kspace, coils, img, fname) in enumerate(loader):
        # Determine dimensions dynamically if not stored in config
        if M is None or N is None:
            _, _, cur_M, cur_N = kspace.shape
            M, N = cur_M, cur_N
            
        mask = np.ones_like(kspace) # TODO: Your dataloader should ideally return the specific sparse mask!
        
        kspace_undersampled_gpu = torch.tensor(kspace * mask, device=device)
        mask_gpu = torch.tensor(mask, device=device)
        coils_gpu = torch.tensor(coils, device=device)
        
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
            img_l_ss=img_l_ss
        )
        
        save_path = os.path.join(save_dir, f"reconstruction_batch{batch_idx}.npy")
        np.save(save_path, reconstruction)
        print(f"Saved: {save_path}")

if __name__ == "__main__":
    main()
