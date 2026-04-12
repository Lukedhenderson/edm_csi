import torch
import torch.nn as nn
import utils

class LoupeMask1d(nn.Module):
    def __init__(
        self,
        shape,
        init_resolution=1,
        force_resolution_mask=False,
        sparsity=0.25,
        slope=10.0,
        fixed_center_radius=0,
        slope2=None,
        straight_through=True,
        inference_state=True,
    ):
        super().__init__()
        if force_resolution_mask:
            shape = [shape[0],shape[1],round(shape[2]*init_resolution)]
        B,M,N = shape[0],shape[1],shape[2]
        self.shape = shape
        self.sparsity = sparsity
        self.slope1 = slope
        self.slope2 = slope2 if slope2 is not None else slope
        self.straight_through = straight_through
        self.inference_state = inference_state
        # init, assuming object has shape of length 3
        logits = torch.randn((B,1,N))*0.5
        if not force_resolution_mask:
            Nr = round(init_resolution*N)
            logits[...,:(N//2-Nr//2)] = -100
            logits[...,(N//2+Nr//2):] = -100
        self.logits = nn.Parameter(logits)

        # create ACS mask if needed
        self.fixed_center_radius = fixed_center_radius
        if fixed_center_radius > 0:
            # square ACS region centered in k-space
            cy = N// 2
            y0 = max(cy - fixed_center_radius, 0)
            y1 = min(cy + fixed_center_radius, N)
            center_mask = torch.zeros((1,1,shape[2]))
            center_mask[..., y0:y1] = 1.0
            self.register_buffer('center_mask', center_mask)
        else:
            self.center_mask = None

    def prob_rescale(self, probs):
        """
        LOUPE-style rescaling to enforce expected sparsity.
        """
        x_bar = torch.mean(probs)
        r = self.sparsity / x_bar
        beta = (1 - self.sparsity) / (1 - x_bar)

        le = (r <= 1).float()
        scaled_probs = le * probs * r + (1 - le) * (1 - (1 - probs) * beta)
        return scaled_probs

    def forward(self):
        # sigmoid to get probabilities from logits
        probs = torch.sigmoid(self.slope1 * self.logits)
        # enforce ACS
        if self.center_mask is not None:
            probs = probs * (1 - self.center_mask) + self.center_mask
        # save pre-rescale probs
        self.last_raw_probs = probs.detach()

        # rescale to target sparsity
        prob_mask = self.prob_rescale(probs)

        # draw random mask ~ U[0,1]
        sample_mask = torch.rand_like(prob_mask)
        if self.center_mask is not None:
            # ensure ACS region is not randomized
            sample_mask = sample_mask * (1 - self.center_mask)

        # create pseudo sampling mask with sigmoid (STE)
        inter = self.slope2 * (prob_mask - sample_mask)
        inter_mask = torch.sigmoid(inter)

        # straight-through binarization with quantile threshold (LOUPE_model.py line 159-167)
        if self.inference_state or self.straight_through:
            thresh_val = torch.quantile(inter_mask, q=(1 - self.sparsity))
            bool_mask = inter_mask >= thresh_val
            binary_mask = bool_mask.to(inter_mask.dtype)
            final_mask = inter_mask + (binary_mask - inter_mask).detach()
        else:
            final_mask = inter_mask
        #m2d = m1d.transpose(-1, -2).expand(-1, 192, 192)
        return prob_mask, final_mask.reshape(1,1,1,self.shape[2]).expand(1,1,self.shape[1],self.shape[2]).squeeze(1)

class LoupeMask(nn.Module):
    """
    Learnable Mask (LOUPE).
    Implements the same rescaling and straight-through
    binarization logic as in LOUPE_MoDL.LOUPE.
    """
    def __init__(
        self,
        shape,
        init_resolution=1,
        force_resolution_mask=False,
        sparsity=0.25,
        slope=10.0,
        fixed_center_radius=0,
        slope2=None,
        straight_through=True,
        inference_state=True,
    ):
        super().__init__()
        if force_resolution_mask:
            shape = [shape[0],round(shape[1]*init_resolution),round(shape[2]*init_resolution)]
        self.shape = shape  # [Nx, Ny] or [1, Nx, Ny] or [ETL, Nx, Ny]
        self.sparsity = sparsity
        # LOUPE uses two slopes: slope1 (for logits->prob) and slope2 (for STE)
        self.slope1 = slope
        self.slope2 = slope2 if slope2 is not None else slope
        self.straight_through = straight_through
        self.inference_state = inference_state

        # init
        logits = torch.randn(shape)*0.5
        if not force_resolution_mask:
            logits = utils.value_from_center(logits,init_resolution,-100)
        self.logits = nn.Parameter(logits)

        # create ACS mask if needed
        self.fixed_center_radius = fixed_center_radius
        if fixed_center_radius > 0:
            # square ACS region centered in k-space
            nx, ny = shape[-2:]
            cx, cy = nx // 2, ny // 2
            x0 = max(cx - fixed_center_radius, 0)
            x1 = min(cx + fixed_center_radius, nx)
            y0 = max(cy - fixed_center_radius, 0)
            y1 = min(cy + fixed_center_radius, ny)
            center_mask = torch.zeros(shape)
            center_mask[..., x0:x1, y0:y1] = 1.0
            self.register_buffer('center_mask', center_mask)
        else:
            self.center_mask = None

    def prob_rescale(self, probs):
        """
        LOUPE-style rescaling to enforce expected sparsity.
        """
        x_bar = torch.mean(probs)
        r = self.sparsity / x_bar
        beta = (1 - self.sparsity) / (1 - x_bar)

        le = (r <= 1).float()
        scaled_probs = le * probs * r + (1 - le) * (1 - (1 - probs) * beta)
        return scaled_probs

    def forward(self):
        """
        Returns:
            prob_mask: rescaled probability mask P (soft, in [0,1])
            final_mask: STE binary-ish mask B used for sampling,
                        matching LOUPE's inter_mask + straight-through.
        """
        # sigmoid to get probabilities from logits
        probs = torch.sigmoid(self.slope1 * self.logits)

        # enforce ACS
        if self.center_mask is not None:
            probs = probs * (1 - self.center_mask) + self.center_mask

        # save pre-rescale probs
        self.last_raw_probs = probs.detach()

        # rescale to target sparsity
        prob_mask = self.prob_rescale(probs)

        # draw random mask ~ U[0,1]
        sample_mask = torch.rand_like(prob_mask)
        if self.center_mask is not None:
            # ensure ACS region is not randomized
            sample_mask = sample_mask * (1 - self.center_mask)

        # create pseudo sampling mask with sigmoid (STE)
        inter = self.slope2 * (prob_mask - sample_mask)
        inter_mask = torch.sigmoid(inter)

        # straight-through binarization with quantile threshold (LOUPE_model.py line 159-167)
        if self.inference_state or self.straight_through:
            thresh_val = torch.quantile(inter_mask, q=(1 - self.sparsity))
            bool_mask = inter_mask >= thresh_val
            binary_mask = bool_mask.to(inter_mask.dtype)
            final_mask = inter_mask + (binary_mask - inter_mask).detach()
        else:
            final_mask = inter_mask

        return prob_mask, final_mask