import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# TODO: remove this dependency
from torchsearchsorted import searchsorted


# Misc
img2mse = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()
        
    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim


# Model
class NeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        """ 
        """
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
        
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Ray helpers
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d


def get_rays_np(H, W, focal, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d
    
    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]
    
    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)
    
    return rays_o, rays_d


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples


######################
# VIEW SPACE HELPERS #
######################

def build_M(n, f, r, t):
    """Compute ndc projection matrix, from view space to ndc.
    Matrix form of `ndc_rays` above.
    Source: https://github.com/bmild/nerf/files/4451808/ndc_derivation.pdf
    """
    if f == np.inf:  # np doesn't do lhopitals
        return np.array([
            [n / r, 0, 0, 0],
            [0, n / t, 0, 0],
            [0, 0, -1, -2 * n],
            [0, 0, -1, 0],
        ])
    return np.array([
        [n / r, 0, 0, 0],
        [0, n / t, 0, 0],
        [0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)],
        [0, 0, -1, 0],
    ])


def to_homogenous(M):
    return M / (M[..., -1:] + 1e-15)


def screen_to_ndc(H, W, focal, c2w, depth):
    """Returns ndc homogenous coordinates"""
    rays_o, rays_d = get_rays_np(H, W, focal, c2w)
    rays_o, rays_d = torch.Tensor(rays_o), torch.Tensor(rays_d)
    rays_o, rays_d = ndc_rays(H, W, focal, 1., rays_o, rays_d)
    pts1_ndc = rays_o + rays_d * depth[..., None]
    pts1_ndc = np.concatenate([pts1_ndc, np.ones_like(pts1_ndc[..., :1])],
                              axis=-1)
    return pts1_ndc


def ndc_to_world(M, ndc):
    """Camera ndc to camera view space
    Defining M s.t. Mv = v' with homogenous world space v and ndc v'.
    As a result, v^T M^T = v'^T and v^T = v'^T (M^T)^-1
    """
    cam = ndc.dot(np.linalg.inv(M.T))
    return cam


def camera1_to_camera2(c2w1, c2w2, pts1_cam):  # unused
    """Camera1 view space to camera2 view space"""
    c2w1_h = np.vstack([c2w1, np.array([0, 0, 0, 1])])
    c2w2_h = np.vstack([c2w2, np.array([0, 0, 0, 1])])
    c12c2_h = c2w1_h.dot(np.linalg.inv(c2w2_h))
    pts2_cam = np.sum(pts1_cam[..., np.newaxis, :] * c12c2_h,
                      -1)  # dot product, equals to: [c12c2_h.dot(pt) for pt in cam]
    pts2_cam = to_homogenous(pts2_cam)
    return pts2_cam


def camera_to_ndc(M, cam):  # unused
    """Camera view space to ndc"""
    ndc = to_homogenous(cam.dot(M.T))
    return ndc


def world_to_screen(cam, c2w, H, W, focal, near=1.0):
    """Inverse of `get_rays`. Returns yx, not xy pairs.
    To sanity check `world_to_screen`, run the following:

        >>> pts1_ndc = screen_to_ndc(H, W, focal, c2w1, depth)  # cam1 screen -> cam1 ndc
        >>> pts1_screen = world_to_screen(pts1_cam, c2w1, H, W, focal, near)

    Expect to see

        >>> pts1_screen[..., 0]
        array([[  0.,   1.,   2., ..., 501., 502., 503.],
            [  0.,   1.,   2., ..., 501., 502., 503.],
            [  0.,   1.,   2., ..., 501., 502., 503.],
            ...,
            [ -0.,   1.,   2., ..., 501., 502., 503.],
            [ -0.,   1.,   2., ..., 501., 502., 503.],
            [ -0.,   1.,   2., ..., 501., 502., 503.]])
        >>> pts1_screen[..., 1]
        array([[378., 378., 378., ..., 378., 378., 378.],
            [377., 377., 377., ..., 377., 377., 377.],
            [376., 376., 376., ..., 376., 376., 376.],
            ...,
            [  3.,   3.,   3., ...,   3.,   3.,   3.],
            [  2.,   2.,   2., ...,   2.,   2.,   2.],
            [  1.,   1.,   1., ...,   1.,   1.,   1.]])
    """
    w2c = np.linalg.inv(c2w[:3, :3])
    rays_o, _ = get_rays_np(H, W, focal, c2w)
    rays_d = cam[..., :-1] - rays_o  # rays from pinhole to point cloud
    dirs = np.sum(rays_d[..., np.newaxis, :] * w2c,
                  -1)  # dot product, equals to: [w2c.dot(dir) for dir in dirs]
    dirs = to_homogenous(dirs)
    d0 = - dirs[..., 0] * focal + W * 0.5
    d1 = dirs[..., 1] * focal + H * 0.5
    screen = np.floor(np.stack([d1, d0], -1))  # NOTE: yx pairs, not xy pairs!
    return screen.astype(int)


def screen_to_world(depth, c2w, H, W, focal, near=1.0, far=np.inf):
    M = build_M(near, far, W / (2.0 * focal), H / (2.0 * focal))  # proj matrix
    ndc = screen_to_ndc(H, W, focal, c2w, depth)  # cam1 screen -> cam1 ndc
    cam = ndc_to_world(M, ndc)  # cam1 ndc -> world
    cam = to_homogenous(cam)
    return cam


def get_cam1_to_cam2_mapping(depth, c2w1, c2w2, H, W, focal, near=1.0,
                             far=np.inf):
    """Get mapping from all camera 1 pixels to all camera 2 pixels.

    To get points in ndc for camera2, use `camera_to_ndc(M, pts2_cam)`. This is not
    needed for the below functionality.
    :return:
        pts2_screen: The returned mapping is (H, W, 2). Say the top-left value, at
                     pixel (0, 0) is (0, 1). This means (0, 0) in camera 1 is sent
                     to pixel (0, 1) in camera 2.
    """
    M = build_M(near, far, W / (2.0 * focal), H / (2.0 * focal))  # proj matrix
    pts = screen_to_world(depth, c2w1, H, W, focal, near,
                          far)  # cam1 depth -> cam1 ndc -> world
    screen = world_to_screen(pts, c2w2, H, W, focal,
                             near)  # world -> cam2 screen
    return screen


#############
# VISUALIZE #
#############

def visualize_point_cloud(points, colors=None):
    assert len(points.shape) == 2 and points.shape[-1] == 3, points.shape
    import open3d as o3d

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud("tmp.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = o3d.io.read_point_cloud("tmp.ply")
    o3d.visualization.draw_geometries([pcd_load])