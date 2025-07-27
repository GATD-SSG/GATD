import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
# import torch.distributed as dist
# from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
from tensorboardX import SummaryWriter

from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from utils.common_util import draw_scalar_field2D
from utils import dist_util

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        tblog_dir = os.path.join(logger.get_current().get_dir(), "tblog")
        self.tb = SummaryWriter(tblog_dir)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size # * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            # if dist.get_rank() == 0:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

        # dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            # if dist.get_rank() == 0:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        # dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()
        
        if self.step % 5000 == 0:
            self._sample_and_visualize()

    def _sample_and_visualize(self):
        print("Sampling and visualizing...")
        self.ddp_model.eval()

        batch, cond = next(self.data)
        _shape = [2] + list(batch.shape[1:])
        with th.no_grad():
            sample = self.diffusion.p_sample_loop(self.ddp_model, _shape, progress=True, model_kwargs=cond, clip_denoised=False)
        sample = sample.detach().cpu().numpy()
        feat_dim = sample.shape[1]
        for i in range(sample.shape[0]):
            fig = draw_scalar_field2D(sample[i, 0])
            self.tb.add_figure(f"sample{i}_c0", fig, global_step=self.step)
            fig = draw_scalar_field2D(sample[i, feat_dim // 2])
            self.tb.add_figure(f"sample{i}_c{feat_dim // 2}", fig, global_step=self.step)

        # if self.step == 0:
        fig = draw_scalar_field2D(batch[0, 0].detach().cpu().numpy())
        self.tb.add_figure("data_c0", fig, global_step=self.step)
        fig = draw_scalar_field2D(batch[0, feat_dim // 2].detach().cpu().numpy())
        self.tb.add_figure(f"data_c{feat_dim // 2}", fig, global_step=self.step)

        self.ddp_model.train()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch.to(dist_util.dev())
            micro_cond = cond
            # micro_cond = {k: v.to(dist_util.dev()) for k, v in cond.items()}
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            self.mp_trainer.backward(loss)

            if self.step % 10 == 0:
                self.log_loss_dict(
                    self.diffusion, t, {k: v * weights for k, v in losses.items()}
                )

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        if self.step % 10 == 0:
            self.tb.add_scalar("step", self.step + self.resume_step, global_step=self.step)
            self.tb.add_scalar("samples", (self.step + self.resume_step + 1) * self.global_batch, global_step=self.step)
            self.tb.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=self.step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            # if dist.get_rank() == 0:
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        # save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        # if dist.get_rank() == 0:
        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

        # dist.barrier()

    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            loss_dict = {}
            logger.logkv_mean(key, values.mean().item())
            loss_dict[f"{key}_mean"] = values.mean().item()
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                loss_dict[f"{key}_q{quartile}"] = sub_loss
            self.tb.add_scalars(f"{key}", loss_dict, global_step=self.step)


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


class MultiScaleTriplaneTrainer:
    """
    多分辨率三平面扩散训练器
    """
    
    def __init__(
        self,
        *,
        model,
        diffusion,
        multiscale_data,
        sizes,
        rescale_losses,
        n_scales,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.multiscale_data = multiscale_data  # 多尺度数据列表
        self.sizes = sizes  # 各尺度的尺寸
        self.rescale_losses = rescale_losses  # 重建损失
        self.n_scales = n_scales
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        tblog_dir = os.path.join(logger.get_current().get_dir(), "tblog")
        self.tb = SummaryWriter(tblog_dir)

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self.sync_cuda = th.cuda.is_available()

        # 创建数据迭代器
        self.data_iters = []
        for i in range(n_scales):
            clear_data, blurry_data = multiscale_data[i]
            if i > 0:
                # 对于高尺度，需要清晰和模糊版本
                data_iter = self._create_multiscale_data_iterator(clear_data, blurry_data, sizes[i], batch_size)
            else:
                # 对于最小尺度，只有清晰版本
                data_iter = self._create_data_iterator(clear_data, sizes[i], batch_size)
            self.data_iters.append(data_iter)

        # 计算训练权重，基于重建损失
        self.scale_weights = th.ones(n_scales)
        if len(rescale_losses) > 0:
            # 使用重建损失计算权重
            loss_weights = th.tensor(rescale_losses)
            self.scale_weights[1:] = loss_weights
            self.scale_weights = self.scale_weights / self.scale_weights.sum()
        else:
            self.scale_weights = self.scale_weights / n_scales

        print(f"尺度训练权重: {self.scale_weights}")

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        self.use_ddp = False
        self.ddp_model = self.model

    def _create_data_iterator(self, triplane_data, size, batch_size):
        """创建单尺度数据迭代器"""
        from utils.triplane_util import compose_featmaps
        feat_xy, feat_xz, feat_yz = triplane_data
        H, W, D = size
        
        # 组合triplane数据
        composed_map, _ = compose_featmaps(feat_xy, feat_xz, feat_yz)
        composed_map = composed_map.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        cond = {'H': H, 'W': W, 'D': D}
        
        while True:
            yield composed_map, cond

    def _create_multiscale_data_iterator(self, clear_data, blurry_data, size, batch_size):
        """创建多尺度数据迭代器（清晰和模糊版本）"""
        from utils.triplane_util import compose_featmaps
        
        # 清晰版本
        clear_xy, clear_xz, clear_yz = clear_data
        clear_composed, _ = compose_featmaps(clear_xy, clear_xz, clear_yz)
        clear_composed = clear_composed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 模糊版本
        blurry_xy, blurry_xz, blurry_yz = blurry_data
        blurry_composed, _ = compose_featmaps(blurry_xy, blurry_xz, blurry_yz)
        blurry_composed = blurry_composed.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        H, W, D = size
        cond = {'H': H, 'W': W, 'D': D}
        
        while True:
            yield (clear_composed, blurry_composed), cond

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                dist_util.load_state_dict(
                    resume_checkpoint, map_location=dist_util.dev()
                )
            )

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = dist_util.load_state_dict(
                ema_checkpoint, map_location=dist_util.dev()
            )
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        """多分辨率训练主循环"""
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # 随机选择尺度进行训练（基于权重）
            scale_idx = th.multinomial(self.scale_weights, 1).item()
            
            # 获取对应尺度的数据
            data_iter = self.data_iters[scale_idx]
            batch_data, cond = next(data_iter)
            
            # 运行训练步骤
            self.run_step(batch_data, cond, scale_idx)
            
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            
        # 保存最终检查点
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_data, cond, scale_idx):
        """运行单个训练步骤"""
        self.forward_backward(batch_data, cond, scale_idx)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step(scale_idx)
        
        if self.step % 5000 == 0:
            self._sample_and_visualize(scale_idx)

    def forward_backward(self, batch_data, cond, scale_idx):
        """前向和反向传播，支持模糊/去模糊操作"""
        self.mp_trainer.zero_grad()
        
        # 处理多尺度数据
        if scale_idx > 0 and isinstance(batch_data, tuple):
            # 高尺度：有清晰和模糊版本
            clear_batch, blurry_batch = batch_data
            clear_batch = clear_batch.to(dist_util.dev())
            blurry_batch = blurry_batch.to(dist_util.dev())
        else:
            # 最小尺度：只有清晰版本
            clear_batch = batch_data.to(dist_util.dev())
            blurry_batch = clear_batch  # 模糊版本等于清晰版本
        
        # 训练
        t, weights = self.schedule_sampler.sample(clear_batch.shape[0], dist_util.dev())
        
        # 在模型kwargs中添加scale信息
        model_kwargs = dict(cond)
        
        # 使用多分辨率扩散的训练损失
        if hasattr(self.diffusion, 'training_losses_multiscale'):
            # 使用支持模糊/去模糊的训练损失
            compute_losses = functools.partial(
                self.diffusion.training_losses_multiscale,
                self.ddp_model,
                clear_batch,    # 清晰版本
                blurry_batch,   # 模糊版本
                t,
                scale_idx,      # 尺度索引
                model_kwargs=model_kwargs,
            )
        else:
            # 回退到传统训练损失
            if hasattr(self.model, 'multiscale') and self.model.multiscale:
                model_kwargs['scale'] = scale_idx
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                clear_batch,
                t,
                model_kwargs=model_kwargs,
            )
        
        losses = compute_losses()
        
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )

        loss = (losses["loss"] * weights).mean()
        self.mp_trainer.backward(loss)

        if self.step % 10 == 0:
            log_losses = {f"scale_{scale_idx}_{k}": v * weights for k, v in losses.items()}
            self.log_loss_dict(self.diffusion, t, log_losses)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self, scale_idx):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("scale", scale_idx)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        logger.logkv("lr", self.opt.param_groups[0]["lr"])
        if self.step % 10 == 0:
            self.tb.add_scalar("step", self.step + self.resume_step, global_step=self.step)
            self.tb.add_scalar("scale", scale_idx, global_step=self.step)
            self.tb.add_scalar("samples", (self.step + self.resume_step + 1) * self.global_batch, global_step=self.step)
            self.tb.add_scalar("lr", self.opt.param_groups[0]["lr"], global_step=self.step)

    def _sample_and_visualize(self, scale_idx):
        """采样和可视化，支持多分辨率采样"""
        print(f"在尺度 {scale_idx} 进行采样和可视化...")
        self.ddp_model.eval()

        with th.no_grad():
            # 尝试使用完整的多分辨率采样
            if hasattr(self.diffusion, 'sample_multiscale_triplane') and scale_idx >= self.n_scales - 2:
                # 只有在高尺度时才使用完整的多分辨率采样
                print(f"  使用完整多分辨率采样...")
                sample = self.diffusion.sample_multiscale_triplane(
                    model=self.ddp_model,
                    batch_size=2,
                    final_size_idx=scale_idx,
                    scale_mul=(1, 1, 1),
                    device=dist_util.dev(),
                    progress=False,  # 训练时不显示进度条
                )
            else:
                # 单尺度采样用于可视化
                _, cond = next(self.data_iters[scale_idx])
                H, W, D = cond['H'], cond['W'], cond['D']
                _shape = [2, self.model.in_channels, H + D, W + D]
                
                model_kwargs = dict(cond)
                if hasattr(self.model, 'multiscale') and self.model.multiscale:
                    model_kwargs['scale'] = scale_idx
                
                if hasattr(self.diffusion, 'p_sample_multiscale'):
                    # 使用多分辨率采样但限定在当前尺度
                    sample = self.diffusion.p_sample_loop(
                        self.ddp_model, _shape, progress=False, 
                        model_kwargs=model_kwargs, clip_denoised=False
                    )
                else:
                    # 回退到传统采样
                    sample = self.diffusion.p_sample_loop(
                        self.ddp_model, _shape, progress=False, 
                        model_kwargs=model_kwargs, clip_denoised=False
                    )
                
        sample = sample.detach().cpu().numpy()
        feat_dim = sample.shape[1]
        for i in range(sample.shape[0]):
            fig = draw_scalar_field2D(sample[i, 0])
            self.tb.add_figure(f"sample_scale{scale_idx}_{i}_c0", fig, global_step=self.step)
            fig = draw_scalar_field2D(sample[i, feat_dim // 2])
            self.tb.add_figure(f"sample_scale{scale_idx}_{i}_c{feat_dim // 2}", fig, global_step=self.step)

        self.ddp_model.train()

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def log_loss_dict(self, diffusion, ts, losses):
        for key, values in losses.items():
            loss_dict = {}
            logger.logkv_mean(key, values.mean().item())
            loss_dict[f"{key}_mean"] = values.mean().item()
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
                loss_dict[f"{key}_q{quartile}"] = sub_loss
            self.tb.add_scalars(f"{key}", loss_dict, global_step=self.step)

