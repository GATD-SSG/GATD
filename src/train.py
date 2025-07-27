from utils.parser_util import train_args, encoding_log_dir, diffusion_log_dir, encoding_feat_path
from utils import dist_util
from utils.common_util import seed_all
import os
import numpy as np


def train_ae(args):
    from encoding.model import ShapeAutoEncoder
    from utils.triplane_util import save_triplane_data, save_multiscale_triplane_data

    print("[Training autoencoder]")

    # seed_all(0)
    # dist_util.setup_dist(args.gpu_id)

    assert args.data_path is not None
    log_dir = encoding_log_dir(args.tag)
    ae_model = ShapeAutoEncoder(log_dir, args)
    ae_model.train(args.data_path)

    feat_maps = ae_model.encode()
    print("feat maps shape:", [fm.shape for fm in feat_maps])
    save_path = encoding_feat_path(args.tag)
    feat_maps_np = [fm.squeeze(0).detach().cpu().numpy() for fm in feat_maps]
    
    # 保存原始单分辨率triplane数据
    save_triplane_data(save_path, feat_maps_np[0], feat_maps_np[1], feat_maps_np[2])
    
    # 创建并保存多分辨率triplane数据金字塔（如果启用）
    if getattr(args, 'enable_multiscale', True):
        print("开始创建多分辨率triplane数据金字塔...")
        multiscale_path = os.path.join(args.tag, "multiscale_triplane")
        scale_factor = getattr(args, 'scale_factor', 1.411)
        sizes, rescale_losses, final_scale_factor, n_scales = save_multiscale_triplane_data(
            multiscale_path, feat_maps_np[0], feat_maps_np[1], feat_maps_np[2], 
            scale_factor=scale_factor
        )
        print(f"多分辨率数据已创建: {n_scales}个尺度")
    else:
        print("多分辨率功能已禁用，只保存单分辨率triplane数据")
    
    # save mesh
    save_dir = os.path.join(log_dir, "rec")
    ae_model.decode_texmesh(save_dir, feat_maps, 256)
    

def train_diffusion(args):
    from utils.triplane_util import load_triplane_data, get_data_iterator, load_multiscale_triplane_data
    from utils.common_util import seed_all
    from diffusion.script_util import create_model_and_diffusion_from_args
    from diffusion.resample import create_named_schedule_sampler
    from diffusion.train_util import TrainLoop, MultiScaleTriplaneTrainer
    from diffusion import logger
    import os

    print("[Training diffusion]")

    # seed_all(0)
    # dist_util.setup_dist(args.gpu_id)

    log_dir = diffusion_log_dir(args.tag)
    logger.configure(dir=log_dir)

    # 检查是否启用多分辨率训练
    enable_multiscale = getattr(args, 'enable_multiscale_diffusion', True) and getattr(args, 'enable_multiscale', True)
    multiscale_training = getattr(args, 'multiscale_training', True)
    
    if enable_multiscale and multiscale_training:
        # 多分辨率训练
        logger.log("使用多分辨率训练模式...")
        
        # 检查多分辨率数据是否存在
        multiscale_path = os.path.join(args.tag, "multiscale_triplane")
        if not os.path.exists(multiscale_path):
            logger.log("多分辨率数据不存在，回退到单分辨率训练...")
            enable_multiscale = False
        else:
            logger.log("加载多分辨率triplane数据...")
            try:
                multiscale_data, sizes, rescale_losses, scale_factor, n_scales = load_multiscale_triplane_data(
                    multiscale_path, device=dist_util.dev()
                )
                logger.log(f"成功加载{n_scales}个尺度的数据")
                logger.log(f"各尺度大小: {sizes}")
                logger.log(f"重建损失: {rescale_losses}")
            except Exception as e:
                logger.log(f"加载多分辨率数据失败: {e}")
                logger.log("回退到单分辨率训练...")
                enable_multiscale = False

    if not enable_multiscale or not multiscale_training:
        # 单分辨率训练（原始方法）
        logger.log("使用单分辨率训练模式...")
        logger.log("creating data loader...")
        src_data, sizes = load_triplane_data(encoding_feat_path(args.tag), device=dist_util.dev())
        data_iter = get_data_iterator(src_data, sizes, args.diff_batch_size)

        logger.log("creating model and diffusion...")
        model, diffusion = create_model_and_diffusion_from_args(args)
        model.to(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

        logger.log("training...")
        TrainLoop(
            model=model,
            diffusion=diffusion,
            data=data_iter,
            batch_size=args.diff_batch_size,
            microbatch=-1,
            lr=args.diff_lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=False,
            use_fp16=args.use_fp16,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.diff_n_iters,
        ).run_loop()
    else:
        # 多分辨率训练
        logger.log("creating multiscale model and diffusion...")
        
        # 确保使用多分辨率模型
        if args.diff_net_type not in ["multiscale_unet"]:
            logger.log(f"自动切换网络类型从 {args.diff_net_type} 到 multiscale_unet")
            args.diff_net_type = "multiscale_unet"
        
        # 导入多分辨率扩散模型创建函数
        from diffusion.script_util import create_multiscale_model_and_diffusion
        
        # 创建多分辨率模型和扩散过程
        model, diffusion = create_multiscale_model_and_diffusion(
            args, n_scales, sizes, rescale_losses
        )
        model.to(dist_util.dev())
        # 将gamma参数移动到与模型相同的设备
        diffusion.to_device(dist_util.dev())
        schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
        
        logger.log(f"创建多分辨率扩散模型: {n_scales}个尺度")
        logger.log(f"启用模糊/去模糊: {getattr(args, 'enable_reblurring', True)}")

        logger.log("starting multiscale training...")
        MultiScaleTriplaneTrainer(
            model=model,
            diffusion=diffusion,
            multiscale_data=multiscale_data,
            sizes=sizes,
            rescale_losses=rescale_losses,
            n_scales=n_scales,
            batch_size=args.diff_batch_size,
            microbatch=-1,
            lr=args.diff_lr,
            ema_rate=args.ema_rate,
            log_interval=args.log_interval,
            save_interval=args.save_interval,
            resume_checkpoint=False,
            use_fp16=args.use_fp16,
            schedule_sampler=schedule_sampler,
            weight_decay=args.weight_decay,
            lr_anneal_steps=args.diff_n_iters,
        ).run_loop()


if __name__ == '__main__':
    args = train_args()
    
    seed_all(0)
    dist_util.setup_dist(args.gpu_id)

    if args.only_enc:
        train_ae(args)
    else:
        if args.enc_log is None:
            train_ae(args)
        train_diffusion(args)
