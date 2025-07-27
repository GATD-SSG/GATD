from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet_triplane import TriplaneUNetModelSmall, TriplaneUNetModelSmallRaw, MultiScaleTriplaneUNet
from .gaussian_diffusion import MultiScaleTriplaneGaussianDiffusion
from utils.parser_util import diffusion_defaults, diffusion_model_defaults, args_to_dict


def create_model_and_diffusion_from_args(args):
    """
    Create model and diffusion from args.
    """
    diffusion = create_gaussian_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    
    if type(args.channel_mult) is str:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))
    if args.diff_net_type == "unet_small":
        model = TriplaneUNetModelSmall(**args_to_dict(args, diffusion_model_defaults().keys()))
    elif args.diff_net_type == "unet_raw":
        model = TriplaneUNetModelSmallRaw(**args_to_dict(args, diffusion_model_defaults().keys()))
    elif args.diff_net_type == "multiscale_unet":
        model_args = args_to_dict(args, diffusion_model_defaults().keys())
        model_args['multiscale'] = getattr(args, 'enable_multiscale', True)
        model = MultiScaleTriplaneUNet(**model_args)
    else:
        raise ValueError(f"Unknown diff_net_type: {args.diff_net_type}")
    return model, diffusion


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    betas = gd.get_named_beta_schedule(noise_schedule, steps)
    if use_kl:
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        loss_type = gd.LossType.RESCALED_MSE
    else:
        loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_multiscale_model_and_diffusion(args, n_scales, sizes, rescale_losses):
    """
    创建多分辨率模型和扩散过程
    
    Args:
        args: 训练参数
        n_scales: 尺度数量
        sizes: 各尺度的尺寸
        rescale_losses: 重建损失列表
    """
    # 创建基础扩散参数，使用create_gaussian_diffusion的参数
    base_diffusion = create_gaussian_diffusion(**args_to_dict(args, diffusion_defaults().keys()))
    
    # 从基础扩散模型中提取参数
    gaussian_diffusion_kwargs = {
        'betas': base_diffusion.betas,
        'model_mean_type': base_diffusion.model_mean_type,
        'model_var_type': base_diffusion.model_var_type,
        'loss_type': base_diffusion.loss_type,
        'rescale_timesteps': base_diffusion.rescale_timesteps,
    }
    
    # 创建多分辨率扩散模型
    diffusion = MultiScaleTriplaneGaussianDiffusion(
        n_scales=n_scales,
        sizes=sizes,
        rescale_losses=rescale_losses,
        reblurring=getattr(args, 'enable_reblurring', True),
        **gaussian_diffusion_kwargs
    )
    
    # 创建模型 
    if type(args.channel_mult) is str:
        args.channel_mult = tuple(int(ch_mult) for ch_mult in args.channel_mult.split(","))
    
    # 根据diff_net_type决定模型类型和multiscale设置
    model_args = args_to_dict(args, diffusion_model_defaults().keys())
    
    if args.diff_net_type == "multiscale_unet":
        # 真正的多分辨率模型
        model_args['multiscale'] = True
        model = MultiScaleTriplaneUNet(**model_args)
    else:
        # 兼容模式：使用MultiScaleTriplaneUNet但设置multiscale=False以兼容单分辨率checkpoint
        model_args['multiscale'] = False
        model = MultiScaleTriplaneUNet(**model_args)
    
    return model, diffusion
