"""
Inference runners for MRI reconstruction models.
"""

import os
import torch
import numpy as np
from datasets.fastmri import create_fastmri_loader
from masks.mask_utils import create_mask
from inference.recon_edm import load_edm_model, run_posterior_sampling

class EDMInferenceRunner:
    """Runner for EDM-based MRI reconstruction inference."""

    def __init__(self, config):
        """
        Initialize the EDM inference runner.

        Args:
            config (dict): Full configuration dictionary
        """
        self.config = config
        self.dataset_config = config.get('dataset', {})
        self.model_config = config.get('model', {})
        self.mask_config = config.get('mask', {})
        self.inference_config = config.get('inference', {})

        # Extract parameters
        self.device = self.model_config.get('device', 'cuda:0')
        self.save_dir = self.inference_config.get('save_dir', 'reconstructions')

        # Initialize components
        self.net = None
        self.t_steps = None
        self.dataset = None

    def setup_model(self):
        """Load and setup the EDM model."""
        path_net = self.model_config.get('path_net', '')
        if not path_net:
            raise ValueError("Model path_net must be specified in config")

        num_steps = self.inference_config.get('num_steps', 300)
        sigma_max = self.inference_config.get('sigma_max', 5.0)
        sigma_min = self.inference_config.get('sigma_min', 0.002)
        rho = self.inference_config.get('rho', 7)

        print("Loading EDM Model...")
        self.net, self.t_steps = load_edm_model(path_net, self.device, num_steps, sigma_max, sigma_min, rho)

    def setup_dataset(self):
        """Setup the dataset loader."""
        print("Loading FastMRI Dataset...")
        dataset_config = self.dataset_config.copy()
        dataset_config['device'] = self.device
        self.dataset = create_fastmri_loader(dataset_config)

    def run_inference(self):
        """Run inference on the dataset."""
        if self.net is None:
            raise RuntimeError("Model not loaded. Call setup_model() first.")
        if self.dataset is None:
            raise RuntimeError("Dataset not loaded. Call setup_dataset() first.")

        # Get dimensions
        M = self.inference_config.get('M', None)
        N = self.inference_config.get('N', None)
        if M is None or N is None:
            C, M, N = self.dataset.get_dimensions()
            if M is None or N is None:
                raise ValueError("Could not determine M and N from dataset. Please specify in config.")

        # Get inference parameters
        seeds = self.inference_config.get('seeds', [0])
        img_likelihood_scale = self.inference_config.get('img_likelihood_scale', 1.0)
        class_label = self.inference_config.get('class_label', None)

        # Create output directory
        os.makedirs(self.save_dir, exist_ok=True)

        print("Starting Reconstruction Loop...")
        results = []

        for batch_idx, (kspace, coils, img, fname) in enumerate(self.dataset):
            # Create undersampling mask
            mask = create_mask(self.mask_config, M, N, self.device)

            # Apply mask to k-space (simulate undersampling)
            kspace_undersampled = kspace * mask
            kspace_undersampled_gpu = torch.tensor(kspace_undersampled, device=self.device)
            mask_gpu = torch.tensor(mask, device=self.device)
            coils_gpu = torch.tensor(coils, device=self.device)

            reconstruction = run_posterior_sampling(
                net=self.net,
                t_steps=self.t_steps,
                kspace_undersampled_gpu=kspace_undersampled_gpu,
                mask_gpu=mask_gpu,
                coils_gpu=coils_gpu,
                device=self.device,
                M=M,
                N=N,
                seeds=seeds,
                class_label=class_label,
                img_l_ss=img_likelihood_scale
            )

            # Save reconstruction
            save_path = os.path.join(self.save_dir, f"reconstruction_batch{batch_idx}.npy")
            np.save(save_path, reconstruction)
            print(f"Saved: {save_path}")

            results.append({
                'batch_idx': batch_idx,
                'fname': fname,
                'save_path': save_path,
                'reconstruction': reconstruction
            })

        return results

def run_edm_inference(config):
    """
    Convenience function to run EDM inference.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list: List of reconstruction results
    """
    runner = EDMInferenceRunner(config)
    runner.setup_model()
    runner.setup_dataset()
    return runner.run_inference()