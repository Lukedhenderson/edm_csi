import torch
import numpy as np

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        ### Yamin code ->
        self.generators = []
        for seed in seeds:
            self.generators.append(torch.Generator(device).manual_seed(int(seed) % (1 << 32)))
    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])
    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)
    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


def value_from_center(x,resolution_fraction,value):
    # center set values exceeding (M*resolution,N*resolution) along last two spatial dimensions
    M = x.shape[-2]
    N = x.shape[-1]
    Mc = int(M * resolution_fraction)
    Nc = int(N * resolution_fraction)
    y0 = (M - Mc) // 2
    y1 = y0 + Mc
    z0 = (N - Nc) // 2
    z1 = z0 + Nc
    out= torch.ones((M, N), device=x.device, dtype=x.dtype)*value
    out[...,y0:y1, z0:z1] = x[...,y0:y1, z0:z1]
    return out


def zero_from_center(x,resolution_fraction):
    # center zero out values exceeding (M*resolution,N*resolution) along last two spatial dimensions
    M = x.shape[-2]
    N = x.shape[-1]
    Mc = int(M * resolution_fraction)
    Nc = int(N * resolution_fraction)
    y0 = (M - Mc) // 2
    y1 = y0 + Mc
    z0 = (N - Nc) // 2
    z1 = z0 + Nc
    mask = torch.zeros((M, N), device=x.device, dtype=x.dtype)
    mask[y0:y1, z0:z1] = 1.0
    return x * mask[None, None, :, :]

def make_img_lr(x,resolution_fraction):
    # return a lower resolution image, with same matrix size, by computing fft, zero'ing out external k-space determined by resolution_fraction, and fft'ing back to image space
    return operators.cartesian.ifftnc(zero_from_center(operators.cartesian.fftnc(x,dims=(-1,-2)),resolution_fraction),dims=(-1,-2))

def make_img_lr_1d(x,resolution_fraction):
    img = operators.cartesian.fftnc(x,dims=(-1,-2))
    N = img.shape[-2]
    Nr = round(N*resolution_fraction)
    img[...,:,:(N//2-Nr//2)] = 0
    img[...,:,(N//2+Nr//2):] = 0
    return operators.cartesian.ifftnc(img,dims=(-1,-2))

def bart_to_python(tensor,device='cpu'):
    # reshaping single-map multicoil data in bart convention (MxNx1xC) to python convention (1 x C x M x N)
    if not isinstance(tensor,torch.Tensor):
        tensor = torch.tensor(tensor,device=device)
    return tensor.permute(2,3,0,1).contiguous()

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def mask_to_traj(mask,Ny_lr,Nz_lr,Ny,Nz):
    coords = torch.nonzero(mask, as_tuple=False)
    offy = (Ny - Ny_lr) // 2
    offz = (Nz - Nz_lr) // 2
    iy = coords[:, 0] + offy
    iz = coords[:, 1] + offz
    ky = (iy - (Ny // 2)) / Ny
    kz = (iz - (Nz // 2)) / Nz
    return torch.stack([ky, kz], dim=1)

def cartesian_traj_2d(Ny, Nz,resolution=1,Ry=1,Rz=1,acs=None,device=None, dtype=torch.float32):
    Ny_lr = int(Ny*resolution)
    Nz_lr = int(Nz*resolution)
    mask = torch.zeros((Ny_lr,Nz_lr))
    ky_idx = torch.arange(0, Ny_lr, Ry, device=device)
    kz_idx = torch.arange(0, Nz_lr, Rz, device=device)
    mask[ky_idx[:, None], kz_idx[None, :]] = 1.0
    if acs:
        mask = add_acs(mask,acs[0],acs[1])
    return mask_to_traj(mask,Ny_lr,Nz_lr,Ny,Nz)

def cartesian_poission(Ny,Nz,R=4,resolution=1,acs=None,device=None,dtype=torch.float32):
    Ny_lr = int(Ny*resolution)
    Nz_lr = int(Nz*resolution)
    mask = torch.tensor(np.real(bart.bart(1,f'poisson -Y {Ny_lr} -Z {Nz_lr} -y {math.sqrt(R)} -z {math.sqrt(R)}')),device=device,dtype=dtype).squeeze()
    if acs:
        mask = add_acs(mask,acs[0],acs[1])
    return mask_to_traj(mask,Ny_lr,Nz_lr,Ny,Nz)

def radial(Ny,Nz,R=4,resolution=1,device=None,dtype=torch.float32):
    # Assuming square image size for now
    Nspokes = int(Ny * 1.57 // R * resolution**2)
    traj = torch.tensor(mrinufft.initialize_2D_radial(Nspokes, Ny).reshape(-1, 2).astype(np.float32)).to(device)
    r = torch.linalg.norm(traj, dim=1)
    keep = r <= 0.5*resolution
    return traj[keep]

def get_trajectory(traj_initialization,M,N,resolution,R,device,acs=None):
    if traj_initialization == 'uniform2d':
        return cartesian_traj_2d(M,N,resolution=resolution,Ry=round(np.sqrt(R)),Rz=round(np.sqrt(R)),acs=acs).to(device)
    if traj_initialization == 'uniform1d':
        Ry = 1
        Rz = int(R)
        return cartesian_traj_2d(M,N,resolution=resolution,Ry=Ry,Rz=Rz,acs=acs).to(device)

    elif traj_initialization == 'poisson':
        return cartesian_poission(M,N,resolution=resolution,R=R,acs=acs).to(device)
    elif traj_initialization == 'radial':
        return radial(M,N,resolution=resolution,R=R).to(device)

def add_acs(mask: torch.Tensor, Am: int, An: int):
    """
    mask: (M, N) torch tensor (0/1)
    Am, An: ACS size in ky, kz
    """
    M, N = mask.shape
    cy, cz = M // 2, N // 2

    y0 = cy - Am // 2
    y1 = y0 + Am
    z0 = cz - An // 2
    z1 = z0 + An

    # clamp to valid range (safety)
    y0 = max(0, y0); y1 = min(M, y1)
    z0 = max(0, z0); z1 = min(N, z1)

    mask = mask.clone()
    mask[y0:y1, z0:z1] = 1.0
    return mask

def save_ckpt(path, epoch, ii, model, optimizer):
    ckpt = dict(
        epoch=epoch,
        iter=ii,
        model_state=model.state_dict(),
        optim_state=optimizer.state_dict(),
        trajectory=model.op.trajectory.detach().cpu(),
        lambda_=float(model.op.lambda_.detach().cpu()),
    )
    torch.save(ckpt, path)

def save_ckpt_loupe(path, epoch, ii, model, optimizer, mask, use_unroll=True):
    if use_unroll:
        lambda_=float(model.op.lambda_.detach().cpu()),
    else:
        lambda_=float(model.lambda_.detach().cpu()),

    ckpt = dict(
        epoch=epoch,
        iter=ii,
        model_state=model.state_dict(),
        optim_state=optimizer.state_dict(),
        lambda_=lambda_,
        mask=mask.detach().cpu().numpy()
    )
    torch.save(ckpt, path)