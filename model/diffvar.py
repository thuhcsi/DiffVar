from functools import partial
from typing import List

import numpy as np
import torch
from torch import nn

from model.diffnet import DiffNet
from model.diffspeech import default, extract, noise_like


class DenoiseDiffusion(nn.Module):
    def __init__(self, timesteps: int, loss_type: str, schedule_type:str,
                 x_max: List[float], x_min: List[float], clip_denoised: bool,
                 beta_min: float=1e-4, beta_max: float=0.01, s: float=8e-3):
        if schedule_type == 'linear':
            betas = np.linspace(beta_min, beta_max, timesteps)
        elif schedule_type == 'cosine':
            steps = timesteps + 1
            x = np.linspace(0, steps, steps)
            alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        else:
            raise NotImplementedError(f"Unknow noise schedule type {schedule_type}")

        self.clip_denoised = clip_denoised

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        # self.K_step = K_step
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        self.register_buffer('x_min', torch.FloatTensor(x_min)[None, None, :])
        self.register_buffer('x_max', torch.FloatTensor(x_max)[None, None, :])
        
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool, denoise_fn=None):
        if denoise_fn is None:
            denoise_fn = self.denoise_fn
        noise_pred = denoise_fn(x, t, cond=cond)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, cond, clip_denoised=True, repeat_noise=False, denoise_fn=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, cond=cond, clip_denoised=clip_denoised, denoise_fn=denoise_fn)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond, noise=None, nonpadding=None, denoise_fn=None):
        if denoise_fn is None:
            denoise_fn = self.denoise_fn
        
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = denoise_fn(x_noisy, t, cond)

        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - x_recon).abs() * nonpadding.unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = (noise - x_recon).abs().mean()

        elif self.loss_type == 'l2':
            # loss = F.mse_loss(noise, x_recon)
            
            if nonpadding is not None:
                loss = (((noise - x_recon)**2) * nonpadding.unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = ((noise - x_recon)**2).mean()
        else:
            raise NotImplementedError()

        return loss

    def norm(self, x):
        return (x - self.x_min) / (self.x_max - self.x_min) * 2 - 1
    
    def denorm(self, x):
        return (x + 1) / 2 * (self.x_max - self.x_min) + self.x_min
        
class DiffVariancePredictor(DenoiseDiffusion):
    def __init__(self, model_config, in_dim=None):
        hparams = model_config["diffusion"]
        self.timestep = hparams["timesteps"]
        self.in_dim = hparams["in_dim"] if in_dim is None else in_dim

        nn.Module.__init__(self)
        DenoiseDiffusion.__init__(
            self,
            timesteps = hparams["timesteps"],
            loss_type=hparams["diff_loss_type"],
            schedule_type=hparams["schedule_type"],
            beta_max=hparams["max_beta"],
            x_max = hparams["x_max"], # [channels]
            x_min = hparams["x_min"], # [channels]
            clip_denoised = hparams["clip_denoised"],
        )

        if isinstance(self.in_dim, int):
            self.total_dim = self.in_dim
            self.in_dim = [list(range(self.in_dim))]
        else:
            self.total_dim = sum([len(chs) for chs in self.in_dim])
            
        self.denoise_fn = nn.ModuleList([DiffNet(
            in_dims = len(chs),
            hidden_size = model_config["transformer"]["decoder_hidden"],
            residual_layers = hparams["residual_layers"],
            residual_channels = hparams["residual_channels"],
            dilation_cycle_length = hparams["dilation_cycle_length"]
        ) for chs in self.in_dim])

    def split_channel(self, x): # ipnut x: [B, 1, C, T]
        return [x[:, :, chs] for chs in self.in_dim]

    def gather_channel(self, x): 
        # TODO: implement proper gather method for non-continuous channel index
        return torch.cat(x, dim=2)

    def get_cond(self, fs2_model, batch):
        speakers, texts, src_lens, max_src_len = batch[:4]
        quasi_symbols = batch[-1]
        
        output, src_lens, max_src_len, src_masks = fs2_model.encode(speakers, texts, src_lens, max_src_len, quasi_symbols)
        
        return output.transpose(1,2), src_masks
    
    def training_step(self, batch, fs2_model):
        with torch.no_grad():
            cond, cond_masks = self.get_cond(fs2_model, batch)

            t = torch.randint(0, self.timestep, (cond.size(0),), device=cond.device)
            
            p_targets = batch[-4]
            e_targets = batch[-3]
            batch[-2][batch[-2]==0] = 1
            log_d_targets = torch.log(batch[-2])
            # d = batch[-2]
            # print(d.size())
            # print(d.max(dim=-1)[0].max(dim=0)[0], d.min(dim=-1)[0].min(dim=0)[0])
            # print(torch.log(d).max(), torch.log(d).min())
            # exit(0)
            var_tgt = self.norm(torch.stack([p_targets, e_targets, log_d_targets], dim=-1))
        var_tgt = var_tgt.transpose(1,2)[:, None, :, :] # [B, 1, M, T]
        
        diff_loss = [
            self.p_losses(v_tgt, t, cond, nonpadding=~cond_masks, denoise_fn=d_fn)
            for d_fn, v_tgt in zip(self.denoise_fn, self.split_channel(var_tgt))
        ]

        return diff_loss

    def validation_step(self, batch, fs2_model):
        with torch.no_grad():
            cond, cond_masks = self.get_cond(fs2_model, batch)

            x = torch.randn((
                cond.shape[0], 1, self.total_dim, cond.shape[2]
            ), device=cond.device)

            res = []
            for d_fn, x_d in zip(self.denoise_fn, self.split_channel(x)):
                for i in reversed(range(0, self.timestep)):
                    x_d = self.p_sample(
                        x_d,
                        torch.full((cond.shape[0],), i, device=cond.device, dtype=torch.long),
                        cond,
                        clip_denoised=self.clip_denoised,
                        denoise_fn=d_fn,
                    )
                res.append(x_d)
            x = self.gather_channel(res)
            
            x = x[:, 0].transpose(1, 2)
            x = self.denorm(x)
            return x, cond, cond_masks
