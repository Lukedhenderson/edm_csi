import torch
import numpy as np
from common.utils import add_acs

def create_cartesian_mask(M, N, acceleration_factor=4, resolution=1.0, acs_size=None, device='cpu'):
    """
    Create a Cartesian undersampling mask for k-space.

    Args:
        M (int): Matrix size in first dimension
        N (int): Matrix size in second dimension
        acceleration_factor (int): Acceleration factor (R)
        resolution (float): Resolution factor
        acs_size (list): [acs_y, acs_z] size of auto-calibration region
        device (str): Device for the mask tensor

    Returns:
        torch.Tensor: Binary mask of shape (1, 1, M, N)
    """
    M_lr = int(M * resolution)
    N_lr = int(N * resolution)

    # Create uniform Cartesian mask
    mask = torch.zeros((M_lr, N_lr), device=device)

    # Sample every acceleration_factor lines
    ky_idx = torch.arange(0, M_lr, acceleration_factor, device=device)
    kz_idx = torch.arange(0, N_lr, acceleration_factor, device=device)

    # Create sampling pattern
    mask[ky_idx[:, None], kz_idx[None, :]] = 1.0

    # Add ACS region if specified
    if acs_size is not None:
        mask = add_acs(mask, acs_size[0], acs_size[1])

    # Expand to full resolution if needed
    if resolution < 1.0:
        # Zero-pad to full size
        mask_full = torch.zeros((M, N), device=device)
        start_y = (M - M_lr) // 2
        start_z = (N - N_lr) // 2
        mask_full[start_y:start_y+M_lr, start_z:start_z+N_lr] = mask
        mask = mask_full

    # Add batch and coil dimensions: (1, 1, M, N)
    return mask.unsqueeze(0).unsqueeze(0)

def create_mask(config, M, N, device='cpu'):
    """
    Create undersampling mask based on configuration.

    Args:
        config (dict): Mask configuration
        M (int): Matrix size in first dimension
        N (int): Matrix size in second dimension
        device (str): Device for the mask tensor

    Returns:
        torch.Tensor: Binary mask of shape (1, 1, M, N)
    """
    mask_type = config.get('type', 'uniform_cartesian')
    acceleration_factor = config.get('acceleration_factor', 4)
    resolution = config.get('resolution', 1.0)
    acs_size = config.get('acs_size', None)

    if mask_type == 'uniform_cartesian':
        return create_cartesian_mask(M, N, acceleration_factor, resolution, acs_size, device)
    elif mask_type == 'full':
        # Full sampling mask
        return torch.ones((1, 1, M, N), device=device)
    else:
        raise ValueError(f"Unsupported mask type: {mask_type}")