import os
from utils.parser_util import sample_args, encoding_feat_path, encoding_log_dir, diffusion_model_path
from utils import dist_util


def sample_diffusion(args):
    from utils.triplane_util import load_triplane_data, decompose_featmaps, save_triplane_data, load_multiscale_triplane_data
    from diffusion.script_util import create_model_and_diffusion_from_args, create_multiscale_model_and_diffusion
    
    # dist_util.setup_dist(args.gpu_id)

    # 检查是否使用多分辨率采样
    enable_multiscale = getattr(args, 'enable_multiscale_sampling', True) and getattr(args, 'enable_multiscale', True)
    multiscale_path = os.path.join(args.tag, "multiscale_triplane")
    
    if enable_multiscale and os.path.exists(multiscale_path):
        print("使用多分辨率采样...")
        
        # 加载多分辨率数据信息
        try:
            multiscale_data, sizes_list, rescale_losses, scale_factor, n_scales = load_multiscale_triplane_data(
                multiscale_path, device=dist_util.dev()
            )
            print(f"加载多分辨率信息: {n_scales}个尺度")
            print(f"各尺度大小: {sizes_list}")
            
            # 首先尝试创建兼容的多分辨率模型
            model_path = diffusion_model_path(args.tag, args.ema_rate, args.diff_n_iters)
            checkpoint = dist_util.load_state_dict(model_path, map_location="cpu")
            
            # 检查checkpoint的时间嵌入维度来判断是否为单分辨率训练的模型
            time_embed_weight_shape = checkpoint.get("time_embed.0.weight", None)
            if time_embed_weight_shape is not None:
                input_dim = time_embed_weight_shape.shape[1]  # 输入维度
                if input_dim == 64:  # 单分辨率模型 (model_channels -> time_embed_dim)
                    print("检测到单分辨率checkpoint，使用兼容模式创建模型")
                    # 创建兼容的多分辨率模型 (multiscale=False)
                    model, diffusion = create_multiscale_model_and_diffusion(args, n_scales, sizes_list, rescale_losses)
                    model.load_state_dict(checkpoint)
                    model.to(dist_util.dev()).eval()
                    diffusion.to_device(dist_util.dev())
                    print("成功加载单分辨率checkpoint到兼容模式")
                    use_legacy_sampling = True  # 标记使用传统采样
                else:
                    print("检测到多分辨率checkpoint")
                    # 创建真正的多分辨率模型 (multiscale=True)
                    args_copy = args.__dict__.copy()
                    temp_args = type('Args', (), args_copy)()
                    temp_args.diff_net_type = "multiscale_unet"
                    model, diffusion = create_multiscale_model_and_diffusion(temp_args, n_scales, sizes_list, rescale_losses)
                    model.load_state_dict(checkpoint)
                    model.to(dist_util.dev()).eval()
                    diffusion.to_device(dist_util.dev())
                    print("成功加载多分辨率checkpoint")
                    use_legacy_sampling = False
            else:
                print("无法检测checkpoint类型，回退到单分辨率采样")
                raise Exception("FALLBACK_TO_SINGLE_SCALE")
            
            # 使用最大尺度的尺寸作为目标
            target_H, target_W, target_D = sizes_list[-1]
            
            # 应用resize参数
            target_H = int(target_H * args.resize[0])
            target_W = int(target_W * args.resize[1])
            target_D = int(target_D * args.resize[2])
            
            print(f"目标采样尺寸: ({target_H}, {target_W}, {target_D})")
            
            result_dir = os.path.join(args.tag, args.output)
            os.makedirs(result_dir, exist_ok=True)
            
            # 获取通道数
            C = model.in_channels if hasattr(model, 'in_channels') else 12
            batch_size = args.diff_batch_size
            
            result_paths = []
            for i in range(0, args.n_samples, batch_size):
                bs = min(batch_size, args.n_samples - i)
                
                print(f"采样批次 {i//batch_size + 1}/{(args.n_samples-1)//batch_size + 1}...")
                
                if use_legacy_sampling:
                    # 单分辨率采样（兼容模式）
                    print("使用兼容模式进行传统单分辨率采样")
                    sample_shape = (bs, C, target_H + target_W + target_D, target_H + target_W + target_D)
                    samples = diffusion.p_sample_loop(
                        model,
                        sample_shape,
                        model_kwargs={"H": target_H, "W": target_W, "D": target_D},
                        device=dist_util.dev(),
                        progress=True,
                    )
                else:
                    # 多分辨率采样
                    print("使用多分辨率采样")
                    samples = diffusion.sample_multiscale_triplane(
                        model=model,
                    batch_size=bs,
                    final_size_idx=-1,  # 使用最大尺寸
                    scale_mul=args.resize,
                    device=dist_util.dev(),
                    progress=True,
                )
                
                # 分解triplane
                samples_xy, samples_xz, samples_yz = decompose_featmaps(samples, (target_H, target_W, target_D))
                samples_xy = samples_xy.detach().cpu().numpy()
                samples_xz = samples_xz.detach().cpu().numpy()
                samples_yz = samples_yz.detach().cpu().numpy()
                
                for j in range(bs):
                    save_path = os.path.join(result_dir, f"{i+j:03d}", "feat.npz")
                    save_triplane_data(save_path, samples_xy[j], samples_xz[j], samples_yz[j])
                    result_paths.append(save_path)
                    
            sampling_type = "兼容模式单分辨率" if use_legacy_sampling else "多分辨率"
            print(f"{sampling_type}采样完成，生成了 {len(result_paths)} 个样本")
            return result_paths
            
        except Exception as e:
            if "FALLBACK_TO_SINGLE_SCALE" in str(e):
                print("由于checkpoint兼容性问题，回退到单分辨率采样...")
            else:
                print(f"多分辨率采样失败: {e}")
                print("回退到单分辨率采样...")
            enable_multiscale = False
    
    if not enable_multiscale:
        print("使用单分辨率采样...")
        
        # 原始单分辨率采样
        src_data, sizes = load_triplane_data(encoding_feat_path(args.tag), device=dist_util.dev())

        model, diffusion = create_model_and_diffusion_from_args(args)
        model_path = diffusion_model_path(args.tag, args.ema_rate, args.diff_n_iters)
        model.load_state_dict(dist_util.load_state_dict(model_path, map_location="cpu"))
        model.to(dist_util.dev()).eval()

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

        result_dir = os.path.join(args.tag, args.output)
        os.makedirs(result_dir, exist_ok=True)

        C = src_data.shape[0]
        H, W, D = sizes
        batch_size = args.diff_batch_size
        H, W, D = int(H * args.resize[0]), int(W * args.resize[1]), int(D * args.resize[2])
        print("H, W, D:", H, W, D)

        result_paths = []
        for i in range(0, args.n_samples, batch_size):
            bs = min(batch_size, args.n_samples - i)
            out_shape = [bs, C, H + D, W + D]

            cond = {'H': H, 'W': W, 'D': D}
            samples = sample_fn(model, out_shape, progress=True, model_kwargs=cond)
            samples_xy, samples_xz, samples_yz = decompose_featmaps(samples, (H, W, D))
            samples_xy = samples_xy.detach().cpu().numpy()
            samples_xz = samples_xz.detach().cpu().numpy()
            samples_yz = samples_yz.detach().cpu().numpy()

            for j in range(bs):
                save_path = os.path.join(result_dir, f"{i+j:03d}", "feat.npz")
                save_triplane_data(save_path, samples_xy[j], samples_xz[j], samples_yz[j])
                result_paths.append(save_path)
        return result_paths


def decode(args, paths):
    from encoding.model import ShapeAutoEncoder
    from utils.triplane_util import load_triplane_data
    import glob

    # dist_util.setup_dist(args.gpu_id)
    
    log_dir = encoding_log_dir(args.tag)
    ae_model = ShapeAutoEncoder(log_dir, args)
    ae_model.load_ckpt("final")

    for path in paths:
        feat_maps = load_triplane_data(path, device=dist_util.dev(), compose=False)
        feat_maps = [fm.unsqueeze(0) for fm in feat_maps]

        save_dir = os.path.dirname(path)
        if args.vox:
            ae_model.decode_voxel(save_dir, feat_maps, args.reso)
        else:
            if args.copy_mtl:
                try:
                    mtl_path = glob.glob(os.path.join(os.path.dirname(args.data_path), "mesh/*.mtl"))[0]
                except:
                    mtl_path = None
            else:
                mtl_path = None
            ae_model.decode_texmesh(save_dir, feat_maps, args.reso, n_faces=args.n_faces, texture_reso=args.texreso,
                                    save_highres_mesh=False, n_surf_pc=-1, mtl_path=mtl_path, file_format=args.file_format)


if __name__ == "__main__":
    args = sample_args()
    dist_util.setup_dist(args.gpu_id)

    result_paths = sample_diffusion(args)
    decode(args, result_paths)
