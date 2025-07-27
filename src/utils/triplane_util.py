import torch
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image
from pathlib import Path


def compose_featmaps(feat_xy, feat_xz, feat_yz):
    H, W = feat_xy.shape[-2:]
    D = feat_xz.shape[-1]

    empty_block = torch.zeros(list(feat_xy.shape[:-2]) + [D, D], dtype=feat_xy.dtype, device=feat_xy.device)
    composed_map = torch.cat(
        [torch.cat([feat_xy, feat_xz], dim=-1),
         torch.cat([feat_yz.transpose(-1, -2), empty_block], dim=-1)], 
        dim=-2
    )
    return composed_map, (H, W, D)


def decompose_featmaps(composed_map, sizes):
    H, W, D = sizes
    feat_xy = composed_map[..., :H, :W] # (C, H, W)
    feat_xz = composed_map[..., :H, W:] # (C, H, D)
    feat_yz = composed_map[..., H:, :W].transpose(-1, -2) # (C, W, D)
    return feat_xy, feat_xz, feat_yz


def pad_composed_featmaps(composed_map, sizes, pad_sizes):
    # pad_sizes: [[padH1, padH2], [padW1, padW2], [padD1, padD2]]
    feat_xy, feat_xz, feat_yz = decompose_featmaps(composed_map, sizes)
    feat_xy = F.pad(feat_xy, pad_sizes[1] + pad_sizes[0])
    feat_xz = F.pad(feat_xz, pad_sizes[2] + pad_sizes[0])
    feat_yz = F.pad(feat_yz, pad_sizes[2] + pad_sizes[1])
    composed_map, new_sizes = compose_featmaps(feat_xy, feat_xz, feat_yz)
    return composed_map, new_sizes


def save_triplane_data(path, feat_xy, feat_xz, feat_yz):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, 
                        feat_xy=feat_xy, feat_xz=feat_xz, feat_yz=feat_yz)


def load_triplane_data(path, device="cuda:0", compose=True):
    data = np.load(path)
    feat_xy = data['feat_xy'][:]
    feat_xz = data['feat_xz'][:]
    feat_yz = data['feat_yz'][:]
    # print("feat_xy shape:", feat_xy.shape)
    # print("feat_xz shape:", feat_xz.shape)
    # print("feat_yz shape:", feat_yz.shape)

    feat_xy = torch.from_numpy(feat_xy).float().to(device) # (C, H, W)
    feat_xz = torch.from_numpy(feat_xz).float().to(device) # (C, H, D)
    feat_yz = torch.from_numpy(feat_yz).float().to(device) # (C, W, D)

    if not compose:
        return feat_xy, feat_xz, feat_yz

    composed_map, (H, W, D) = compose_featmaps(feat_xy, feat_xz, feat_yz)
    return composed_map, (H, W, D)


def get_data_iterator(featmaps_data, sizes, batch_size=1):
    featmaps_data = featmaps_data.unsqueeze(0).expand(batch_size, -1, -1, -1)
    H, W, D = sizes

    while True:
        yield featmaps_data, {'H': H, 'W': W, 'D': D}


def create_triplane_scales(feat_xy, feat_xz, feat_yz, scale_factor=1.411, create_dirs=True, base_path=None):

    C, H, W = feat_xy.shape
    D = feat_xz.shape[-1]

    area_scale_0 = 3110
    min_dim = min(H, W, D)
    scale_0_dim = int(round(np.sqrt(area_scale_0 * min_dim / max(H, W, D))))
    scale_0_dim = max(16, min(scale_0_dim, 32))

    if H < feat_xy.shape[1] or W < feat_xy.shape[2] or D < feat_xz.shape[2]:
        print(f"缩放特征图从 ({feat_xy.shape[1]}, {feat_xy.shape[2]}, {feat_xz.shape[2]}) 到 ({H}, {W}, {D})")
        feat_xy_tensor = torch.from_numpy(feat_xy).unsqueeze(0)
        feat_xz_tensor = torch.from_numpy(feat_xz).unsqueeze(0)
        feat_yz_tensor = torch.from_numpy(feat_yz).unsqueeze(0)
        
        feat_xy = F.interpolate(feat_xy_tensor, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).numpy()
        feat_xz = F.interpolate(feat_xz_tensor, size=(H, D), mode='bilinear', align_corners=False).squeeze(0).numpy()
        feat_yz = F.interpolate(feat_yz_tensor, size=(W, D), mode='bilinear', align_corners=False).squeeze(0).numpy()

    #n_scales = int(round((np.log(min_dim / scale_0_dim)) / (np.log(scale_factor))) + 1)
    #scale_factor = np.exp((np.log(min_dim / scale_0_dim)) / (n_scales - 1))

    n_scales = 5
    
    print(f"创建{n_scales}个尺度的triplane金字塔，scale_factor={scale_factor:.3f}")
    
    sizes = []
    downscaled_triplanes = []
    recon_triplanes = []
    rescale_losses = []
    
    # 创建各个尺度的数据
    for i in range(n_scales):
        # 计算当前尺度的尺寸
        cur_scale = np.power(scale_factor, n_scales - i - 1)
        cur_H = int(round(H / cur_scale))
        cur_W = int(round(W / cur_scale))
        cur_D = int(round(D / cur_scale))
        
        # 确保尺寸是偶数
        cur_H = max(4, cur_H // 2 * 2)
        cur_W = max(4, cur_W // 2 * 2)
        cur_D = max(4, cur_D // 2 * 2)

        # cur_H = H
        # cur_W = W
        # cur_D = D
        
        print(f"尺度 {i}: ({cur_H}, {cur_W}, {cur_D})")
        
        # 将numpy数组转为tensor进行插值
        feat_xy_tensor = torch.from_numpy(feat_xy).unsqueeze(0)  # (1, C, H, W)
        feat_xz_tensor = torch.from_numpy(feat_xz).unsqueeze(0)  # (1, C, H, D)
        feat_yz_tensor = torch.from_numpy(feat_yz).unsqueeze(0)  # (1, C, W, D)
        
        # 下采样到当前尺度
        cur_feat_xy = F.interpolate(feat_xy_tensor, size=(cur_H, cur_W), mode='bilinear', align_corners=False)
        cur_feat_xz = F.interpolate(feat_xz_tensor, size=(cur_H, cur_D), mode='bilinear', align_corners=False)
        cur_feat_yz = F.interpolate(feat_yz_tensor, size=(cur_W, cur_D), mode='bilinear', align_corners=False)
        
        # cur_feat_xy = feat_xy_tensor
        # cur_feat_xz = feat_xz_tensor
        # cur_feat_yz = feat_yz_tensor

        # 转回numpy
        cur_feat_xy = cur_feat_xy.squeeze(0).numpy()
        cur_feat_xz = cur_feat_xz.squeeze(0).numpy()
        cur_feat_yz = cur_feat_yz.squeeze(0).numpy()
    
        downscaled_triplanes.append((cur_feat_xy, cur_feat_xz, cur_feat_yz))
        sizes.append((cur_H, cur_W, cur_D))
        
        # 保存清晰版本
        if create_dirs and base_path:
            scale_dir = os.path.join(base_path, f'scale_{i}')
            os.makedirs(scale_dir, exist_ok=True)
            save_triplane_data(
                os.path.join(scale_dir, 'triplane.npz'),
                cur_feat_xy, cur_feat_xz, cur_feat_yz
            )
    
    # 创建重建（模糊）版本
    for i in range(n_scales - 1):
        # 将更小尺度的数据上采样到当前尺度
        smaller_xy, smaller_xz, smaller_yz = downscaled_triplanes[i + 1]
        target_H, target_W, target_D = sizes[i]
        
        # 上采样
        smaller_xy_tensor = torch.from_numpy(smaller_xy).unsqueeze(0)
        smaller_xz_tensor = torch.from_numpy(smaller_xz).unsqueeze(0)
        smaller_yz_tensor = torch.from_numpy(smaller_yz).unsqueeze(0)
        
        recon_feat_xy = F.interpolate(smaller_xy_tensor, size=(target_H, target_W), mode='bilinear', align_corners=False)
        recon_feat_xz = F.interpolate(smaller_xz_tensor, size=(target_H, target_D), mode='bilinear', align_corners=False)
        recon_feat_yz = F.interpolate(smaller_yz_tensor, size=(target_W, target_D), mode='bilinear', align_corners=False)

        # recon_feat_xy = smaller_xy_tensor
        # recon_feat_xz = smaller_xz_tensor
        # recon_feat_yz = smaller_yz_tensor
        
        recon_feat_xy = recon_feat_xy.squeeze(0).numpy()
        recon_feat_xz = recon_feat_xz.squeeze(0).numpy()
        recon_feat_yz = recon_feat_yz.squeeze(0).numpy()
        
        recon_triplanes.append((recon_feat_xy, recon_feat_xz, recon_feat_yz))
        
        # 计算重建损失 - 使用相对误差而不是绝对误差
        orig_xy, orig_xz, orig_yz = downscaled_triplanes[i]

        mse_xy = np.mean((orig_xy - recon_feat_xy) ** 2)
        mse_xz = np.mean((orig_xz - recon_feat_xz) ** 2)
        mse_yz = np.mean((orig_yz - recon_feat_yz) ** 2)
        
        # 归一化到[0, 1]范围
        var_xy = np.var(orig_xy)
        var_xz = np.var(orig_xz)
        var_yz = np.var(orig_yz)
        
        rel_loss_xy = mse_xy / (var_xy + 1e-8)
        rel_loss_xz = mse_xz / (var_xz + 1e-8)
        rel_loss_yz = mse_yz / (var_yz + 1e-8)
        
        avg_loss = (rel_loss_xy + rel_loss_xz + rel_loss_yz) / 3
        
        # 限制在合理范围内
        avg_loss = np.clip(avg_loss, 0.001, 1.0)
        rescale_losses.append(avg_loss)
        
        print(f"尺度 {i} 重建损失: {avg_loss:.6f} (MSE: xy={mse_xy:.6f}, xz={mse_xz:.6f}, yz={mse_yz:.6f})")
        
        # 保存模糊版本
        if create_dirs and base_path:
            recon_dir = os.path.join(base_path, f'scale_{i}_recon')
            os.makedirs(recon_dir, exist_ok=True)
            save_triplane_data(
                os.path.join(recon_dir, 'triplane.npz'),
                recon_feat_xy, recon_feat_xz, recon_feat_yz
            )
    
    return sizes, rescale_losses, scale_factor, n_scales


def save_multiscale_triplane_data(base_path, feat_xy, feat_xz, feat_yz, scale_factor=1.411):
    """
    保存多尺度的triplane数据
    """
    print("开始创建多分辨率三平面数据金字塔...")
    
    sizes, rescale_losses, final_scale_factor, n_scales = create_triplane_scales(
        feat_xy, feat_xz, feat_yz, 
        scale_factor=scale_factor,
        create_dirs=True,
        base_path=base_path
    )
    
    # 保存元数据
    meta_data = {
        'sizes': sizes,
        'rescale_losses': rescale_losses,
        'scale_factor': final_scale_factor,
        'n_scales': n_scales
    }
    
    meta_path = os.path.join(base_path, 'meta.npz')
    np.savez_compressed(meta_path, **meta_data)
    
    print(f"多分辨率数据已保存到: {base_path}")
    print(f"尺度数量: {n_scales}, 最终scale_factor: {final_scale_factor:.3f}")
    print(f"各尺度大小: {sizes}")
    print(f"重建损失: {rescale_losses}")
    
    return sizes, rescale_losses, final_scale_factor, n_scales


def load_multiscale_triplane_data(base_path, device="cuda:0"):
    """
    加载多尺度triplane数据
    """
    # 加载元数据
    meta_path = os.path.join(base_path, 'meta.npz')
    meta_data = np.load(meta_path)
    sizes = meta_data['sizes']
    rescale_losses = meta_data['rescale_losses']
    scale_factor = float(meta_data['scale_factor'])
    n_scales = int(meta_data['n_scales'])
    
    # 加载各尺度数据
    scales_data = []
    for i in range(n_scales):
        scale_dir = os.path.join(base_path, f'scale_{i}')
        scale_path = os.path.join(scale_dir, 'triplane.npz')
        
        # 清晰版本
        clear_data = load_triplane_data(scale_path, device=device, compose=False)
        
        if i == 0:
            # 最小尺度：只有清晰版本，模糊版本等于清晰版本
            scales_data.append((clear_data, clear_data))
        elif i < n_scales - 1:
            # 中间尺度：应该有重建版本
            recon_dir = os.path.join(base_path, f'scale_{i}_recon')
            recon_path = os.path.join(recon_dir, 'triplane.npz')
            
            if os.path.exists(recon_path):
                blurry_data = load_triplane_data(recon_path, device=device, compose=False)
            else:
                print(f"警告：预期的重建版本 {recon_path} 不存在，使用清晰版本")
                blurry_data = clear_data
            
            scales_data.append((clear_data, blurry_data))
        else:
            # 最大分辨率：设计上没有重建版本，模糊版本等于清晰版本
            print(f"最大分辨率尺度 {i}：使用清晰版本作为模糊版本")
            scales_data.append((clear_data, clear_data))
    
    return scales_data, sizes, rescale_losses, scale_factor, n_scales
