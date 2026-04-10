#%reset -f
# Reconstructing with EDM
# --- EXTRACTED: essential imports for posterior sampling ---
import sys, os, re, pathlib, pickle, tqdm, torch, numpy as np
import json, argparse

# Add the 'edm' submodule to the Python path so we can import its dependencies
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'edm'))
import dnnlib, utils
from torch_utils import distributed as dist

from fastMRI_dataloader import loader #change if using your own dataloader


tnpy = lambda x: x.cpu().detach().numpy()
recon_to_numpy = lambda reconstruction_gpu: torch.view_as_complex(reconstruction_gpu.permute(0,-2,-1,1).contiguous())[None].cpu().numpy().squeeze()
# --- COMMENTED OUT: project-specific imports ---
# import loupe.loupe, bart, cfl
# os.environ['DEBUG_LEVEL']='0'
# from loaders.loaders import KspSensImgLoader
# from torch.utils.data import DataLoader
# from sigpy.plot import ImagePlot as iplt
# cat = np.concatenate
# --- COMMENTED OUT: project-specific paths ---
# # - Paths
# data_path = '/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/datasets/fastmri_brain_white_standardsize_v0_num_coils12/val'
# model_tik_rough = 2500 # which model tik we want to roughly use
# edm_model_base_path = '/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/edm_models/fastmri_brain_white_standardsize_v0_num_coils12_res0.9_dim2' # also specifies the resolution
# save_base_path = '/csiNAS3/yarefeen/accelerated_recon_hypothesis/src/edm_recons'
# --- Load Configuration ---
parser = argparse.ArgumentParser(description="EDM MRI Reconstruction")
parser.add_argument('--config', type=str, default='recon_config.json', help='Path to the configuration file')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = json.load(f)

path_net = config.get('path_net', '')
device = config.get('device', 'cuda:0')
save_dir = config.get('save_dir', 'reconstructions')

# - Posterior sampling params
seeds = config.get('seeds', [0])
num_steps = config.get('num_steps', 300)
img_lss = config.get('img_lss', 1.0)
mot_lss = config.get('mot_lss', 1.0)
sigma_max = config.get('sigma_max', 5.0)
sigma_min = config.get('sigma_min', 0.002)
S_noise = config.get('S_noise', 0)
motion_est = config.get('motion_est', 0)
rho = config.get('rho', 7)
img_l_ss = config.get('img_l_ss', 1.0)
class_label = config.get('class_label', None)

# --- COMMENTED OUT: project-specific experiment parameters ---
# # - Parameters
# total_accelerations = [8,12,16,20,24,28,32]
# acs = 12
# dimension = 2 # 1D or 2D under-sampling
# num_val = 50
# --- COMMENTED OUT: project-specific misc/mask params ---
# # - Misc params
# shuffle = False
# batch_size = 1
# num_workers = 0
# device = 'cuda:2'
# # - Mask params
# slope1 = 1.0
# slope2 = 10


M = config.get('M', None)  # image height
N = config.get('N', None)  # image width
# --- COMMENTED OUT: project-specific model checkpoint finder ---
# # - loading edm model
# pattern = re.compile(r"network-snapshot-(\d{6})\.pkl$")
# matches = []
# for f in pathlib.Path(edm_model_base_path).rglob("network-snapshot-*.pkl"):
#     m = pattern.match(f.name)
#     if m:
#         step = int(m.group(1))
#         matches.append((abs(step - model_tik_rough), step, f))
# _, _ , closest_path = min(matches, key=lambda x: (x[0], x[1]))
# path_net = str(closest_path)
# --- EXTRACTED: load EDM model from checkpoint ---
with dnnlib.util.open_url(path_net, verbose=(dist.get_rank() == 0)) as f:
    net = pickle.load(f)['ema'].to(device)
# --- COMMENTED OUT: project-specific resolution extraction ---
# # - Getting resolution from file name
# resolution = float((re.search(r'res([0-9.]+)', edm_model_base_path)).group(1))
# --- EXTRACTED: calculate posterior sampling time step schedule ---
step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
# --- COMMENTED OUT: project-specific acceleration loop, mask generation, data iteration ---
# # Looping through accelerations
# # Storing
# all_recs = []
# all_refs = []
# all_masks = []
# for aa, total_acceleration in enumerate(total_accelerations):
#     if dimension == 2:
#         R = total_acceleration*resolution**2
#     if dimension == 1:
#         R = total_acceleration*resolution
#     sparsity = 1/R
#     acs_res = acs*resolution**2
#     # - getting under-sampling mask from loupe operator
#     if dimension == 2:
#         loupe_mask = loupe.loupe.LoupeMask(shape=shape,init_resolution=resolution,force_resolution_mask=True,sparsity=sparsity, slope=slope1, slope2=slope2, fixed_center_radius=acs//2,straight_through=True,inference_state=True).to(device)
#     elif dimension == 1:
#         loupe_mask = loupe.loupe.LoupeMask1d(shape=shape,init_resolution=resolution,force_resolution_mask=True,sparsity=sparsity, slope=slope1, slope2=slope2, fixed_center_radius=acs//2,straight_through=True,inference_state=True).to(device)
#     with torch.no_grad():
#         _,mask = loupe_mask()
#     # - padding mask
#     pad_M = shape[1] - mask.shape[1]
#     pad_N = shape[2] - mask.shape[2]
#     pad_top    = pad_M // 2
#     pad_bottom = pad_M - pad_top
#     pad_left   = pad_N // 2
#     pad_right  = pad_N - pad_left
#     mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom))
#     mask = tnpy(mask)
#     all_masks.append(mask)
#     # - to append
#     refs = []
#     recs = []
#     pbar = tqdm.tqdm(loader, desc=f"acceleration {aa+1}/{len(total_accelerations)}")
#     for ii, (kspace, coils, img, fname) in enumerate(pbar):
#         # - data loader
#         kspace = tnpy(kspace).squeeze()
#         coils = tnpy(coils).squeeze()
#         img = tnpy(img).squeeze()
#         # - normalizing img associated with fully-sampled k-space to have norm 1 based on these coils
#         ref = bart.bart(1,'pics -S',kspace[None].transpose(2,3,0,1),coils[None].transpose(2,3,0,1))
#         ref = ref / np.max(abs(ref))
#         refs.append(ref)
#         kspace_normalized = bart.bart(1,'fft -u 6',ref[None]*coils)
#         # - preparing for reconstruction
#         kspace_undersampled_gpu = torch.tensor(kspace_normalized*mask,device=device)[None]
#         mask_gpu = torch.tensor(mask[None],device=device)
#         coils_gpu = torch.tensor(coils[None],device=device)
#         # - to append
#         recs_seeds = []

# --- EXTRACTED: core Diffusion Posterior Sampling (DPS) loop ---
# TODO: provide your own kspace_undersampled_gpu, mask_gpu, coils_gpu tensors on device
kspace_undersampled_gpu = None  # [1, C, M, N] undersampled k-space
mask_gpu = None                 # [1, 1, M, N] undersampling mask
coils_gpu = None                # [1, C, M, N] coil sensitivities

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
    
    # --- Save output ---
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"reconstruction_seed{seed}.npy")
    np.save(save_path, reconstruction)
    print(f"Saved reconstruction to: {save_path}")

# --- COMMENTED OUT: rest of project-specific data/result collection ---
#         recs_seeds.append(recon_to_numpy(x_next))
#         recs.append(recs_seeds)
#         if ii+1 == num_val:
#             # break once reached num_val
#             break
#     all_recs.append(recs)
#     all_refs.append(refs)
# --- COMMENTED OUT: project-specific result aggregation and saving ---
# all_refs = np.array(all_refs)
# all_recs = np.array(all_recs)
# all_masks = np.array(all_masks)
# # Saving
# save_path = path_net.split('/')
# save_path = os.path.join(save_base_path,save_path[-3]+'_'+save_path[-1])
# os.makedirs(save_path, exist_ok=True)
# cfl.writecfl(os.path.join(save_path,'recs'),all_recs)
# cfl.writecfl(os.path.join(save_path,'refs'),all_refs)
# cfl.writecfl(os.path.join(save_path,'masks'),all_masks)
# os.system(f'cp recon_edm.py {save_path}')