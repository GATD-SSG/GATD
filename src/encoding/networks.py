import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DecoderMLP, ResnetBlock, TriplaneGroupResnetBlock, DecoderMLPSkipConcat
import numpy as np
import itertools

def get_networks(cfg):
    print("Encoding network type: {}".format(cfg.enc_net_type))
    use_tex = cfg.data_type != "sdf"
    tex_channels = 8 if cfg.data_type == "sdfpbr" else 3
    if cfg.enc_net_type == "base":
        return AutoEncoderGroupV3(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels, posenc=0)
    elif cfg.enc_net_type == "skip":
        return AutoEncoderGroupSkip(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels, posenc=0)
    elif cfg.enc_net_type == "pbr":
        return AutoEncoderGroupPBR(cfg.fdim_geo, cfg.fdim_tex, cfg.fdim_up, cfg.hidden_dim, cfg.n_hidden_layers, use_tex=use_tex, tex_channels=tex_channels, posenc=0)
    else:
        raise ValueError("Unknown net type: {}".format(cfg.net_type))


class AutoEncoderGroupV3(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )
        self.geo_decoder = DecoderMLP(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers)

        if use_tex:
            self.tex_convs = TriplaneGroupResnetBlock(
                tex_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
            )
            self.tex_decoder = DecoderMLP(feat_channel_up, tex_channels, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]

        #print("feat_map : " + feat_map.shape + "  !!!!")
        #print("x : " + x.shape + "  !!!!")

        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]
        
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            for i in range(3):
                h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])

        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            h_tex = self.tex_decoder(h_tex).sigmoid() # (N, 1)
            h = torch.cat([h_geo, h_tex], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)

def sample_triplanes(xyz, planes):
    """
    xyz: (B, 3)
    planes: dict with keys 'xy', 'yz', 'xz', each with tensor (B, C, H, W)
    returns: (B, C_total)
    """
    B, _ = xyz.shape
    device = xyz.device

    def sample_plane(plane, coords):
        # coords: (B, 2) in [-1, 1]
        #grid = coords.view(B, 1, 1, 2)  # (B, 1, 1, 2)
        grid = coords.view(1, 1, B, 2)  # (1, 1, B, 2)

        #sample_coords = x.view(1, 1, N, 2)
        #feat = F.grid_sample(feat_map, sample_coords.flip(-1),
        #                       align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)

        return F.grid_sample(plane, grid.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)

    #print("planes :    !!!!")
    #print(planes.shape)

    f_xy = sample_plane(planes[0], xyz[:, [0, 1]])
    f_xz = sample_plane(planes[1], xyz[:, [0, 2]])
    f_yz = sample_plane(planes[2], xyz[:, [1, 2]])

    feat = f_xy + f_xz + f_yz

    #return torch.cat([f_xy, f_xz, f_yz], dim=-1)
    return feat

def compute_offset_magnitude(offsets):
    """Compute mean and std of offset magnitudes."""
    magnitudes = np.linalg.norm(offsets, axis=-1)  # (N, K)
    mean = magnitudes.mean()
    std = magnitudes.std()
    return mean, std

def compute_anisotropy_ratio(offsets):
    """
    For each set of K offsets, compute 3x3 covariance matrix,
    then return mean ratio of largest/smallest eigenvalues.
    """
    #N, K, _ = offsets.shape

    ratios = []
    for i in range(1):
        cov = np.cov(offsets[i].T)  # shape (3, 3)
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals <= 1e-8):  # avoid numerical issues
            continue
        ratio = eigvals[-1] / eigvals[0]
        ratios.append(ratio)
    return np.mean(ratios)

def compute_spatial_entropy(offsets, bins=32):
    """
    Convert offsets to angular bins (e.g., spherical coordinates),
    then compute entropy over angular histogram.
    """
    vectors = offsets.reshape(-1, 8)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
    unit_vectors = vectors / norms

    # Convert to spherical coordinates
    x, y, z = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    theta = np.arccos(np.clip(z, -1, 1))  # [0, π]
    phi = np.arctan2(y, x)  # [-π, π]
    phi = (phi + 2 * np.pi) % (2 * np.pi)  # [0, 2π]

    hist, _ = np.histogram2d(theta, phi, bins=[bins, bins], range=[[0, np.pi], [0, 2 * np.pi]])
    hist = hist.flatten()
    hist = hist / hist.sum() + 1e-10  # normalize and avoid log(0)
    return entropy(hist, base=np.e)

def generate_fixed_pattern(K=8, cube_size=0.1):
    """
    Generate K fixed offsets within a cube centered at the origin.

    Parameters:
    - K: number of fixed spatial points.
    - cube_size: edge length of the cube.

    Returns:
    - offsets: (K, 3) numpy array of fixed 3D offsets.
    """
    # Generate points from a 3D grid
    grid_res = int(round(K ** (1 / 3)))
    linspace = np.linspace(-cube_size / 2, cube_size / 2, grid_res)
    grid_points = list(itertools.product(linspace, repeat=3))

    # If we have more than K points, randomly pick K
    if len(grid_points) > K:
        np.random.seed(44)  # for reproducibility
        grid_points = np.random.choice(len(grid_points), size=K, replace=False)
        offsets = np.array([grid_points[i] for i in grid_points])
    else:
        offsets = np.array(grid_points[:K])

    return offsets  # shape (K, 3)

class SpatialPatternPredictor(nn.Module):
    def __init__(self, hidden_dim=64, k=8):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)  # predict k offset vectors
        )

    def forward(self, xyz):
        """
        xyz: (B, 3) - input 3D point
        returns: (B, k, 3) - spatial offsets from the input point
        """

        offsets = self.mlp(xyz)  # (B, k * 3)
        np_offsets = offsets.detach().view(-1, self.k, 1)

        return xyz.unsqueeze(1) + np_offsets

class PointFeatureLifter(nn.Module):
    def __init__(self, feature_dim=64, hidden_dim=128, num_heads=4, agg='mean'):
        super().__init__()
        self.agg = agg
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project features to queries, keys, values
        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output refinement
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

    def forward(self, lifted_feats):
        """
        neighborhood_feats: (B, k, F) - features from pattern-sampled points
        returns: (B, out_dim)
        """
        #if self.agg == 'mean':
        #    agg_feat = neighborhood_feats.mean(dim=1)  # (B, F)
        #elif self.agg == 'max':
        #    agg_feat, _ = neighborhood_feats.max(dim=1)
        #else:
        #    raise ValueError("Unsupported aggregation")

        #return self.mlp(agg_feat)

        B, K, C = lifted_feats.shape
        # Project query
        query = self.q_proj(lifted_feats[:, K // 2])  # use center lifted feature as query (or learnable)
        query = query.unsqueeze(1)  # (B, 1, H)

        keys = self.k_proj(lifted_feats)  # (B, K, H)
        values = self.v_proj(lifted_feats)  # (B, K, H)

        # Attention: Q = (B, 1, H), K/V = (B, K, H)
        attn_output, _ = self.attn(query, keys, values)  # (B, 1, H)
        attn_output = attn_output.squeeze(1)  # (B, H)

        # Output refinement
        fused_feat = self.ffn(attn_output)  # (B, C)

        return fused_feat


class GeoAwareTriplaneFeatureExtractor(nn.Module):
    def __init__(self, triplane_feat_dim, out_dim, k=8):
        super().__init__()
        self.k = k
        self.spp = SpatialPatternPredictor(k=k)
        self.lifter = PointFeatureLifter(triplane_feat_dim)

    def forward(self, xyz, planes):
        """
        xyz: (B, 3)
        planes: dict of triplane features (each B, C, H, W)
        returns: (B, out_dim)
        """
        B = xyz.shape[0]
        pts = self.spp(xyz)  # (B, k, 3)
        feats = []

        for i in range(self.k):
            pt_i = pts[:, i, :]  # (B, 3)
            feat_i = sample_triplanes(pt_i, planes)  # (B, C*3)
            feats.append(feat_i)

        neighborhood_feats = torch.stack(feats, dim=1)  # (B, k, C*3)
        lifted = self.lifter(lifted_feats=neighborhood_feats)
        return lifted

class AutoEncoderGroupSkip(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )

        print('positional encoder = ' + str(posenc))
        self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)

        print('use 3d aware model ! ')
        self.geo_3d_aware_model = GeoAwareTriplaneFeatureExtractor(triplane_feat_dim=feat_channel_up, out_dim=feat_channel_up, k=8)
        self.tex_3d_aware_model = GeoAwareTriplaneFeatureExtractor(triplane_feat_dim=feat_channel_up, out_dim=feat_channel_up, k=8)

        if use_tex:
            self.tex_convs = TriplaneGroupResnetBlock(
                tex_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
            )
            self.tex_decoder = DecoderMLPSkipConcat(feat_channel_up, tex_channels, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + list(self.tex_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]


        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)

        #print("geo_feat_maps!!!!!!!!!!!!!!")
        #print(geo_feat_maps[0].shape)

        h_geo = self.geo_3d_aware_model(x, geo_feat_maps)
        #for i in range(3):
        #    h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            #print("tex_feat_maps!!!!!!!!!!!!!!")
            #print(tex_feat_maps[0].shape)
            #tex_feat_maps = self.tex_3d_aware_model(x, tex_feat_maps)
            #for i in range(3):
            #    h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])
            h_tex = self.tex_3d_aware_model(x, tex_feat_maps)


        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            #h_tex = self.tex_3d_aware_model(x, h_tex)

            h_tex = self.tex_decoder(h_tex).sigmoid() # (N, 1)
            h = torch.cat([h_geo, h_tex], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)


class AutoEncoderGroupPBR(nn.Module):
    def __init__(self, geo_feat_channels, tex_feat_channels, feat_channel_up, mlp_hidden_channels, mlp_hidden_layers, use_tex=True, tex_channels=3, posenc=0) -> None:
        super().__init__()
        self.use_tex = use_tex

        self.geo_encoder = nn.Conv3d(1, geo_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        if use_tex:
            self.tex_encoder = nn.Conv3d(tex_channels + 1, tex_feat_channels, kernel_size=4, stride=2, padding=1, bias=True)
        out_channels = geo_feat_channels + tex_feat_channels if use_tex else geo_feat_channels
        self.norm = nn.InstanceNorm2d(out_channels)

        self.geo_feat_dim = geo_feat_channels
        self.tex_feat_dim = tex_feat_channels

        self.geo_convs = TriplaneGroupResnetBlock(
            geo_feat_channels, feat_channel_up, ks=5, input_norm=False, input_act=False
        )
        self.geo_decoder = DecoderMLPSkipConcat(feat_channel_up, 1, mlp_hidden_channels, mlp_hidden_layers)

        if use_tex:
            self.tex_convs = nn.Sequential(
                TriplaneGroupResnetBlock(tex_feat_channels, feat_channel_up, ks=3, input_norm=False, input_act=False),
                TriplaneGroupResnetBlock(feat_channel_up, feat_channel_up, ks=3, input_norm=True, input_act=True),
            )
            self.rgb_decoder = DecoderMLPSkipConcat(feat_channel_up, 3, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
            self.mr_decoder = DecoderMLPSkipConcat(feat_channel_up, 2, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
            self.normal_decoder = DecoderMLPSkipConcat(feat_channel_up, 3, mlp_hidden_channels, mlp_hidden_layers, posenc=posenc)
        
        self.register_buffer("aabb", torch.tensor([-1, -1, -1, 1, 1, 1], dtype=torch.float32))

    def geo_parameters(self):
        return list(self.geo_encoder.parameters()) + list(self.geo_convs.parameters()) + list(self.geo_decoder.parameters())
    
    def tex_parameters(self):
        return list(self.tex_encoder.parameters()) + list(self.tex_convs.parameters()) + \
              list(self.rgb_decoder.parameters()) + list(self.mr_decoder.parameters()) + list(self.normal_decoder.parameters())

    def reset_aabb(self, aabb):
        print("set net aabb:", aabb)
        if not isinstance(aabb, torch.Tensor):
            aabb = torch.tensor(aabb, dtype=torch.float32)
        # self.register_buffer("aabb", aabb.to(self.encoder.weight.device))
        self.aabb = aabb.to(self.geo_encoder.weight.device)

    def encode(self, vol):
        geo_feat = self.geo_encoder(vol[:, :1])
        if self.use_tex:
            tex_feat = self.tex_encoder(vol)
            vol_feat = torch.cat([geo_feat, tex_feat], dim=1)
        else:
            vol_feat = geo_feat

        xy_feat = vol_feat.mean(dim=4)
        xz_feat = vol_feat.mean(dim=3)
        yz_feat = vol_feat.mean(dim=2)
        
        xy_feat = (self.norm(xy_feat) * 0.5).tanh()
        xz_feat = (self.norm(xz_feat) * 0.5).tanh()
        yz_feat = (self.norm(yz_feat) * 0.5).tanh()

        return [xy_feat, xz_feat, yz_feat]
    
    def sample_feature_plane2D(self, feat_map, x):
        """Sample feature map at given coordinates"""
        # feat_map: [1, C, H, W]
        # x: [N, 2]
        N = x.shape[0]
        sample_coords = x.view(1, 1, N, 2)
        feat = F.grid_sample(feat_map, sample_coords.flip(-1),
                               align_corners=False, padding_mode='border')[0, :, 0, :].transpose(0, 1)
        return feat

    def decode(self, x, feat_maps, aabb=None):
        # x [N, 3]
        if aabb is None:
            aabb = self.aabb
        x = 2 * (x - aabb[:3]) / (aabb[3:] - aabb[:3]) - 1 # [-1, 1]

        h_geo = 0
        h_tex = 0

        coords_list = [[0, 1], [0, 2], [1, 2]]
        
        geo_feat_maps = [fm[:, :self.geo_feat_dim] for fm in feat_maps]
        geo_feat_maps = self.geo_convs(geo_feat_maps)
        for i in range(3):
            h_geo += self.sample_feature_plane2D(geo_feat_maps[i], x[..., coords_list[i]]) # (N, C)

        if self.use_tex:
            tex_feat_maps = [fm[:, self.geo_feat_dim:] for fm in feat_maps]
            tex_feat_maps = self.tex_convs(tex_feat_maps)
            for i in range(3):
                h_tex += self.sample_feature_plane2D(tex_feat_maps[i], x[..., coords_list[i]])

        h_geo = self.geo_decoder(h_geo) # (N, 1)
        if self.use_tex:
            h_rgb = self.rgb_decoder(h_tex) # (N, 3)
            h_mr = self.mr_decoder(h_tex) # (N, 2)
            h_normal = self.normal_decoder(h_tex) # (N, 3)
            h = torch.cat([h_geo, h_rgb, h_mr, h_normal], dim=1)
        else:
            h = h_geo
        return h
    
    def forward(self, vol, x, aabb=None):
        feat_map = self.encode(vol)
        return self.decode(x, feat_map, aabb=aabb)

