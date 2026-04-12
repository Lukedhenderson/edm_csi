#%reset -f
# Reconstructing with EDM
import sys, os, re, pathlib, pickle, tqdm, torch, numpy as np

# Add the 'edm' submodule to the Python path so we can import its dependencies
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edm'))
import dnnlib, utils
from torch_utils import distributed as dist

tnpy = lambda x: x.cpu().detach().numpy()
recon_to_numpy = lambda reconstruction_gpu: torch.view_as_complex(reconstruction_gpu.permute(0,-2,-1,1).contiguous())[None].cpu().numpy().squeeze()

def load_edm_model(path_net, device, num_steps, sigma_max, sigma_min, rho):
    with dnnlib.util.open_url(path_net, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    return net, t_steps

def run_posterior_sampling(net, t_steps, kspace_undersampled_gpu, mask_gpu, coils_gpu, device, M, N, seeds, class_label, img_l_ss):
    """Core Diffusion Posterior Sampling (DPS) loop"""
    for seed in seeds:
        rnd = utils.utils.StackedRandomGenerator(device, [seed])
        latents = rnd.randn([1, 2, M, N], device=device)
        x_next = latents.to(torch.float64) * t_steps[0]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_cur = x_next
            x_hat = x_cur
            x_hat = x_hat.requires_grad_()
            denoised = net(x_hat, t_cur, class_label).to(torch.float64)
            d_cur = (x_hat - denoised)/t_cur
            x_next = x_hat + (t_next - t_cur) * d_cur
            denoised_cplx = torch.view_as_complex(denoised.permute(0,-2,-1,1).contiguous())[None]
            Ax = torch.fft.ifftshift(denoised_cplx*coils_gpu, dim=(-2, -1))
            Ax = torch.fft.fft2(Ax, dim=(-2,-1), norm='ortho')
            Ax = torch.fft.fftshift(Ax, dim=(-2, -1))
            Ax = Ax * mask_gpu
            residual = kspace_undersampled_gpu - Ax
            sse = torch.norm(residual)**2
            likelihood_score = torch.autograd.grad(outputs=sse, inputs=x_hat)[0]
            x_next = x_next - (img_l_ss / torch.sqrt(sse)) * likelihood_score
            x_next = x_next.detach()
            x_hat = x_hat.requires_grad_(False)
    
        reconstruction = recon_to_numpy(x_next)
    return reconstruction