"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
import torch.nn.functional as F

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from utils.triplane_util import decompose_featmaps


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self._scale_timesteps(t), **model_kwargs)
        new_mean = (
            p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        )
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(
            x, self._scale_timesteps(t), **model_kwargs
        )

        out = p_mean_var.copy()
        out["pred_xstart"] = self._predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out

    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
                yield out
                img = out["sample"]

    def ddim_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
        y0=None,
        mask=None,
        is_mask_t0=False,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # masked generation
        if y0 is not None and mask is not None:
            assert y0.shape == x.shape
            assert mask.shape == x.shape
            if is_mask_t0:
                out["pred_xstart"] = mask * y0 + (1 - mask) * out["pred_xstart"]
            else:
                nonzero_mask = (
                    (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
                )  # no noise when t == 0
                out["pred_xstart"] = (mask * y0 + (1 - mask) * out["pred_xstart"]) * nonzero_mask + out["pred_xstart"] * (1 - nonzero_mask)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # Equation 12.
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        y0=None,
        mask=None,
        is_mask_t0=False,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
            y0=y0,
            mask=mask,
            is_mask_t0=is_mask_t0,
        ):
            final = sample
        return final["sample"]

    def ddim_sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
        y0=None,
        mask=None,
        is_mask_t0=False,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    y0=y0,
                    mask=mask,
                    is_mask_t0=is_mask_t0,
                )
                yield out
                img = out["sample"]

    def _vb_terms_bpd(
        self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        sample_coords = sample_coords.repeat(32, 1, 1, 1)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def generate_random_2d_points_tensor(self, x, N, integer=False, device=None):

        _, _, H, W = x.shape
        device = device or x.device

        if integer:
            ys = th.randint(0, H, (N,), device=device)
            xs = th.randint(0, W, (N,), device=device)
        else:
            ys = th.rand(N, device=device) * H  # [0, H)
            xs = th.rand(N, device=device) * W  # [0, W)

        points = th.stack((ys, xs), dim=1)  # 形状 [N, 2]，每行是 (y, x)
        return points

    def compute_mmd_loss(self, f_real, f_gen, kernel='rbf', sigma=1.0):
        """
        f_real, f_gen: (B, N, D)
        """

        def gaussian_kernel(x, y, sigma):
            x = x.unsqueeze(2)  # (B, N, 1, D)
            y = y.unsqueeze(1)  # (B, 1, N, D)
            dist = th.sum((x - y) ** 2, dim=-1)  # (B, N, N)
            return th.exp(-dist / (2 * sigma ** 2))

        K_xx = gaussian_kernel(f_real, f_real, sigma).mean()
        K_yy = gaussian_kernel(f_gen, f_gen, sigma).mean()
        K_xy = gaussian_kernel(f_real, f_gen, sigma).mean()
        return K_xx + K_yy - 2 * K_xy

    def training_losses(self, model, x_start, t, model_kwargs=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            raise NotImplementedError
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs)

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            H, W, D = model_kwargs["H"], model_kwargs["W"], model_kwargs["D"]
            target_xy, target_xz, target_yz = decompose_featmaps(target, (H, W, D))
            model_output_xy, model_output_xz, model_output_yz = decompose_featmaps(model_output, (H, W, D))

            x_start_xy, x_start_xz, x_start_yz = decompose_featmaps(x_start, (H, W, D))

            # model_output 和 x_start 分别采样 n 个点

            pts_xy = self.generate_random_2d_points_tensor(model_output_xy, 262144)
            pts_xz = self.generate_random_2d_points_tensor(model_output_xz, 262144)
            pts_yz = self.generate_random_2d_points_tensor(model_output_yz, 262144)

            f_mo_xy = self.sample_feature_plane2D(model_output_xy, pts_xy)
            f_mo_xz = self.sample_feature_plane2D(model_output_xy, pts_xz)
            f_mo_yz = self.sample_feature_plane2D(model_output_xy, pts_yz)

            f_xs_xy = self.sample_feature_plane2D(x_start_xy, pts_xy)
            f_xs_xz = self.sample_feature_plane2D(x_start_xz, pts_xz)
            f_xs_yz = self.sample_feature_plane2D(x_start_yz, pts_yz)

            terms["loss_xy"] = self.compute_mmd_loss(f_xs_xy, f_mo_xy)
            terms["loss_xz"] = self.compute_mmd_loss(f_xs_xz, f_mo_xz)
            terms["loss_yz"] = self.compute_mmd_loss(f_xs_yz, f_mo_yz)

            terms["mse_xy"] = mean_flat((target_xy - model_output_xy) ** 2)
            terms["mse_xz"] = mean_flat((target_xz - model_output_xz) ** 2)
            terms["mse_yz"] = mean_flat((target_yz - model_output_yz) ** 2)
            # terms["mse"] = mean_flat((target - model_output) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse_xy"] + terms["mse_xz"] + terms["mse_yz"] + terms["vb"] + terms["loss_xy"] + terms["loss_xz"] + terms["loss_yz"]
                # terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse_xy"] + terms["mse_xz"] + terms["mse_yz"] + terms["loss_xy"] + terms["loss_xz"] + terms["loss_yz"]
                # terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class MultiScaleTriplaneGaussianDiffusion(GaussianDiffusion):
    """
    多分辨率三平面高斯扩散模型
    """

    def __init__(self, n_scales, sizes, rescale_losses, reblurring=True, **kwargs):
        # 先调用父类构造函数
        super().__init__(**kwargs)

        # 然后设置子类属性
        self.n_scales = n_scales
        self.sizes = sizes
        self.rescale_losses = rescale_losses
        self.reblurring = reblurring

        # 初始化gamma参数
        self._init_gamma_parameters()

    def _init_gamma_parameters(self):
        self.gammas = []
        if self.reblurring and len(self.rescale_losses) > 0:
            for i, loss in enumerate(self.rescale_losses):
                max_gamma = min(0.5, max(0.05, float(loss) * 50.0))  # 更保守的映射

                timesteps = th.arange(self.num_timesteps).float()
                gamma_tensor = max_gamma * (timesteps / (self.num_timesteps - 1))

                self.gammas.append(gamma_tensor)
                print(f"尺度 {i+1} 的gamma范围: [0.0000, {max_gamma:.4f}] (基于重建损失: {loss:.6f})")
        else:
            # 如果没有重建损失，使用默认的gamma模式
            for i in range(self.n_scales - 1):
                # 默认的时间相关gamma
                max_gamma = 0.5  # 默认最大值
                timesteps = th.arange(self.num_timesteps).float()
                gamma_tensor = max_gamma * (timesteps / (self.num_timesteps - 1))
                self.gammas.append(gamma_tensor)
                print(f"尺度 {i+1} 的默认gamma范围: [0.0000, {max_gamma:.4f}]")

        print(f"初始化了 {len(self.gammas)} 个时间相关的gamma参数")

    def to_device(self, device):
        """将gamma参数移动到指定设备"""
        for i in range(len(self.gammas)):
            self.gammas[i] = self.gammas[i].to(device)
        print(f"Gamma参数已移动到设备: {device}")

    def extract_gamma(self, gammas, t, x_shape):
        """提取时间步对应的gamma值"""
        # 确保gamma参数与时间步t在同一设备上
        if hasattr(t, 'device'):
            gammas = gammas.to(t.device)

        # 直接处理tensor，避免numpy转换
        res = gammas[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def q_sample_with_blur(self, x_start, x_blur, t, scale_idx, noise=None):
        """
        在给定时间步采样，同时考虑噪声和模糊
        """
        if noise is None:
            noise = th.randn_like(x_start)

        # 如果启用模糊且不是最小尺度
        if self.reblurring and scale_idx > 0:
            # 获取当前尺度的gamma值（与时间步相关）
            gamma = self.extract_gamma(self.gammas[scale_idx - 1], t, x_start.shape)

            x_mixed = gamma * x_blur + (1 - gamma) * x_start

            # 对混合后的版本添加噪声
            x_noisy = self.q_sample(x_mixed, t, noise=noise)

            return x_noisy
        else:
            # 最小尺度：只添加噪声，不添加模糊
            return self.q_sample(x_start, t, noise=noise)

    def predict_start_from_noise_with_deblur(self, x_t, t, scale_idx, noise, x_blur_curr=None):
        """
        从噪声预测x_start，同时进行去模糊操作
        """
        # 传统的去噪预测，得到混合版本（如果有模糊的话）
        x_recon_mixed = _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

        # 如果不启用模糊或是最小尺度，直接返回（没有模糊成分）
        if not self.reblurring or scale_idx == 0:
            return x_recon_mixed, x_recon_mixed

        # 去模糊操作：从混合版本中分离出清晰版本
        if x_blur_curr is not None:
            gamma = self.extract_gamma(self.gammas[scale_idx - 1], t, x_recon_mixed.shape)

            epsilon = 1e-8  # 避免除零
            x_deblurred = (x_recon_mixed - gamma * x_blur_curr) / (1 - gamma + epsilon)

            # 返回去模糊后的清晰版本和原始混合版本
            return x_deblurred, x_recon_mixed
        else:
            # 如果没有提供模糊版本，直接返回预测结果
            return x_recon_mixed, x_recon_mixed

    def training_losses_multiscale(self, model, x_start, x_blur, t, scale_idx, model_kwargs=None, noise=None):
        """
        多尺度训练损失，包含模糊操作
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # 使用模糊采样
        x_t = self.q_sample_with_blur(x_start, x_blur, t, scale_idx, noise=noise)

        terms = {}

        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 添加尺度信息到model_kwargs
            model_kwargs_with_scale = dict(model_kwargs)
            if hasattr(model, 'multiscale') and model.multiscale:
                model_kwargs_with_scale['scale'] = scale_idx

            model_output = model(x_t, self._scale_timesteps(t), **model_kwargs_with_scale)

            if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    terms["vb"] *= self.num_timesteps / 1000.0

            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,  # 目标始终是清晰版本
                ModelMeanType.EPSILON: noise,    # 噪声也是清晰版本的噪声
            }[self.model_mean_type]

            assert model_output.shape == target.shape == x_start.shape

            # 分别计算三平面的损失
            from utils.triplane_util import decompose_featmaps
            H, W, D = model_kwargs["H"], model_kwargs["W"], model_kwargs["D"]
            target_xy, target_xz, target_yz = decompose_featmaps(target, (H, W, D))
            model_output_xy, model_output_xz, model_output_yz = decompose_featmaps(model_output, (H, W, D))

            terms["mse_xy"] = mean_flat((target_xy - model_output_xy) ** 2)
            terms["mse_xz"] = mean_flat((target_xz - model_output_xz) ** 2)
            terms["mse_yz"] = mean_flat((target_yz - model_output_yz) ** 2)

            if "vb" in terms:
                terms["loss"] = terms["mse_xy"] + terms["mse_xz"] + terms["mse_yz"] + terms["vb"]
            else:
                terms["loss"] = terms["mse_xy"] + terms["mse_xz"] + terms["mse_yz"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def p_mean_variance_multiscale(self, model, x, t, scale_idx, x_blur_curr=None,
                                   clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        多尺度预测均值和方差，包含去模糊操作
        """
        if model_kwargs is None:
            model_kwargs = {}

        # 添加尺度信息
        model_kwargs_with_scale = dict(model_kwargs)
        if hasattr(model, 'multiscale') and model.multiscale:
            model_kwargs_with_scale['scale'] = scale_idx

        B, C = x.shape[:2]
        assert t.shape == (B,)
        model_output = model(x, self._scale_timesteps(t), **model_kwargs_with_scale)

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            else:
                # 正确的多分辨率采样：使用去模糊逻辑
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )

                # 在多分辨率采样中应用去模糊
                if self.reblurring and scale_idx > 0 and x_blur_curr is not None:
                    # 应用去模糊公式
                    gamma = self.extract_gamma(self.gammas[scale_idx - 1], t, pred_xstart.shape)
                    epsilon = 1e-8
                    pred_xstart_deblurred = (pred_xstart - gamma * x_blur_curr) / (1 - gamma + epsilon)
                    pred_xstart = process_xstart(pred_xstart_deblurred)

            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def p_sample_multiscale(self, model, x, t, scale_idx, x_blur_curr=None,
                           clip_denoised=True, denoised_fn=None, cond_fn=None, model_kwargs=None):
        """
        多尺度采样单步，包含去模糊
        """
        out = self.p_mean_variance_multiscale(
            model, x, t, scale_idx, x_blur_curr=x_blur_curr,
            clip_denoised=clip_denoised, denoised_fn=denoised_fn, model_kwargs=model_kwargs
        )
        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        if cond_fn is not None:
            out["mean"] = self.condition_mean(
                cond_fn, out, x, t, model_kwargs=model_kwargs
            )
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop_multiscale(
        self,
        model,
        final_shape,
        start_scale=0,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        scale_mul=(1, 1, 1),
    ):
        """
        多分辨率采样循环
        """
        if device is None:
            device = next(model.parameters()).device
        if model_kwargs is None:
            model_kwargs = {}

        batch_size = final_shape[0]
        channels = final_shape[1]

        # 应用缩放因子
        final_H_D = int(final_shape[2] * scale_mul[0])  # H + D
        final_W_D = int(final_shape[3] * scale_mul[1])  # W + D

        # 计算最终的H, W, D（假设D相等）
        final_H = int(self.sizes[-1][0] * scale_mul[0])
        final_W = int(self.sizes[-1][1] * scale_mul[1])
        final_D = int(self.sizes[-1][2] * scale_mul[2])

        print(f"多分辨率采样: 目标尺寸 ({final_H}, {final_W}, {final_D})")

        samples_from_scales = []

        for scale_idx in range(start_scale, self.n_scales):
            print(f"采样尺度 {scale_idx}/{self.n_scales-1}")

            # 计算当前尺度的尺寸
            if scale_idx < len(self.sizes):
                # 使用预定义的尺寸
                base_H, base_W, base_D = self.sizes[scale_idx]
            else:
                # 如果超出预定义尺寸，使用最大尺寸
                base_H, base_W, base_D = self.sizes[-1]

            # 应用缩放因子（只在最终尺度应用）
            if scale_idx == self.n_scales - 1:
                # 最终尺度应用缩放
                cur_H = int(base_H * scale_mul[0])
                cur_W = int(base_W * scale_mul[1])
                cur_D = int(base_D * scale_mul[2])
            else:
                # 中间尺度保持原始比例
                cur_H, cur_W, cur_D = base_H, base_W, base_D

            print(f"  当前尺度尺寸: ({cur_H}, {cur_W}, {cur_D})")

            # 当前尺度的shape
            cur_shape = [batch_size, channels, cur_H + cur_D, cur_W + cur_D]

            # 准备model_kwargs
            cur_model_kwargs = dict(model_kwargs)
            cur_model_kwargs.update({'H': cur_H, 'W': cur_W, 'D': cur_D})
            if hasattr(model, 'multiscale') and model.multiscale:
                cur_model_kwargs['scale'] = scale_idx

            if scale_idx == start_scale:
                # 最小尺度：从纯噪声开始完整采样
                print(f"  从噪声开始完整扩散采样...")
                if noise is not None and scale_idx == 0:
                    # 如果提供了噪声且是第一个尺度，调整噪声尺寸
                    init_noise = th.randn(*cur_shape, device=device)
                else:
                    init_noise = None

                sample = self.p_sample_loop(
                    model,
                    cur_shape,
                    noise=init_noise,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=cur_model_kwargs,
                    device=device,
                    progress=progress,
                )
            else:
                # 高尺度：从上一尺度上采样的结果开始
                prev_sample = samples_from_scales[scale_idx - start_scale - 1]

                print(f"  从上一尺度上采样: {prev_sample.shape} -> {cur_shape}")

                # 上采样上一尺度的结果
                upsampled = self._upsample_triplane(prev_sample, cur_H, cur_W, cur_D)

                # 计算注入时间步
                inject_timestep = self._calculate_inject_timestep(scale_idx)
                print(f"  注入时间步: {inject_timestep}")

                # 从注入时间步开始采样
                sample = self._inject_and_sample(
                    model,
                    upsampled,
                    inject_timestep,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=cur_model_kwargs,
                    device=device,
                    progress=progress,
                    scale_idx=scale_idx,
                )

            samples_from_scales.append(sample)
            print(f"  尺度 {scale_idx} 采样完成: {sample.shape}")

        # 返回最高分辨率的结果
        final_sample = samples_from_scales[-1]
        print(f"多分辨率采样完成: {final_sample.shape}")

        return final_sample

    def _upsample_triplane(self, triplane, target_H, target_W, target_D):
        """
        精确上采样triplane到目标尺寸
        """
        from utils.triplane_util import decompose_featmaps, compose_featmaps

        # 当前尺寸
        B, C = triplane.shape[:2]
        current_total_H = triplane.shape[2]  # H + D
        current_total_W = triplane.shape[3]  # W + D

        if hasattr(self, 'sizes') and len(self.sizes) > 0:
            best_match = None
            min_diff = float('inf')
            for h, w, d in self.sizes:
                total_h = h + d
                total_w = w + d
                diff = abs(total_h - current_total_H) + abs(total_w - current_total_W)
                if diff < min_diff:
                    min_diff = diff
                    best_match = (h, w, d)

            if best_match:
                current_H, current_W, current_D = best_match
            else:
                current_D = min(current_total_H, current_total_W) // 3
                current_H = current_total_H - current_D
                current_W = current_total_W - current_D
        else:
            current_D = min(current_total_H, current_total_W) // 3
            current_H = current_total_H - current_D
            current_W = current_total_W - current_D

        try:
            triplane_xy, triplane_xz, triplane_yz = decompose_featmaps(triplane, (current_H, current_W, current_D))

            upsampled_xy = F.interpolate(triplane_xy, size=(target_H, target_W), mode='bilinear', align_corners=False)
            upsampled_xz = F.interpolate(triplane_xz, size=(target_H, target_D), mode='bilinear', align_corners=False)
            upsampled_yz = F.interpolate(triplane_yz, size=(target_W, target_D), mode='bilinear', align_corners=False)

            upsampled, _ = compose_featmaps(upsampled_xy, upsampled_xz, upsampled_yz)

        except Exception as e:
            print(f"精确上采样失败: {e}, 使用简单上采样")
            upsampled = F.interpolate(
                triplane,
                size=(target_H + target_D, target_W + target_D),
                mode='bilinear',
                align_corners=False
            )

        return upsampled

    def _calculate_inject_timestep(self, scale_idx):
        """
        计算注入时间步
        高尺度使用较小的时间步，不需要完整的扩散过程
        """
        if scale_idx == 0:
            return self.num_timesteps - 1

        inject_ratio = 0.8 ** scale_idx
        inject_timestep = int(self.num_timesteps * inject_ratio)
        inject_timestep = max(inject_timestep, self.num_timesteps // 4)  # 最少1/4的步数

        return inject_timestep

    def _inject_and_sample(
        self,
        model,
        x_start,
        inject_timestep,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        scale_idx=0,
    ):
        """
        从给定的x_start在指定时间步注入噪声，然后进行采样
        """
        batch_size = x_start.shape[0]

        # 在inject_timestep添加噪声
        t_inject = th.tensor([inject_timestep] * batch_size, device=device)
        noise = th.randn_like(x_start)
        x_t = self.q_sample(x_start, t_inject, noise=noise)

        # 从t_inject开始反向采样
        indices = list(range(inject_timestep + 1))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices, desc=f"Scale {scale_idx} sampling")

        img = x_t

        for i in indices:
            t = th.tensor([i] * batch_size, device=device)
            with th.no_grad():
                if hasattr(self, 'p_sample_multiscale'):
                    out = self.p_sample_multiscale(
                        model,
                        img,
                        t,
                        scale_idx,
                        x_blur_curr=None,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                else:
                    out = self.p_sample(
                        model,
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                img = out["sample"]

        return img

    def sample_multiscale_triplane(
        self,
        model,
        batch_size=1,
        final_size_idx=-1,  # 最终尺寸索引，默认为最大尺寸
        scale_mul=(1, 1, 1),
        device=None,
        progress=True,
        **kwargs
    ):
        """
        便捷的多分辨率triplane采样函数
        """
        if device is None:
            device = next(model.parameters()).device

        # 确定最终尺寸
        if final_size_idx == -1:
            final_size_idx = len(self.sizes) - 1

        final_H, final_W, final_D = self.sizes[final_size_idx]

        # 应用缩放因子
        final_H = int(final_H * scale_mul[0])
        final_W = int(final_W * scale_mul[1])
        final_D = int(final_D * scale_mul[2])

        # 计算最终形状（假设第一个尺寸来确定通道数）
        channels = model.in_channels if hasattr(model, 'in_channels') else 12  # 默认12通道
        final_shape = [batch_size, channels, final_H + final_D, final_W + final_D]

        print(f"开始多分辨率triplane采样...")
        print(f"目标尺寸: ({final_H}, {final_W}, {final_D})")
        print(f"最终形状: {final_shape}")

        # 执行多分辨率采样
        sample = self.p_sample_loop_multiscale(
            model=model,
            final_shape=final_shape,
            start_scale=0,
            scale_mul=scale_mul,
            device=device,
            progress=progress,
            **kwargs
        )

        return sample
