#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch.nn import functional as F
import math
from gsplat import rasterization
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from lbs.lbs import lrs, pose_conditioned_deform
from utils_loc.sh_utils import eval_sh
import numpy as np


def render_feature(viewpoint_camera,
           pc : GaussianModel,
           bg_color : torch.Tensor,
           scaling_modifier = 1.0,
           override_color = None,
           stage='pose_conditioned',
           render_features = False,
           render_gaussian_idx = False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )


    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

   
    joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    if stage == 'pose_conditioned':
        if pc.args.k_plane: # using k-plane deformation model for deformation instead of linear blend skinning
            means3D, scales, rotations, opacity, shs = pc._deformation(means3D, scales, rotations, opacity, shs, times_sel=None, joints=joints)
            rotations = pc.rotation_activation(rotations)
        else: # use lbs for means+rotations, + MLP for appearance
            means3D_deformed, rotations_deformed = lrs(joints[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=True, lrs_model=pc.lrs_model, rotations=rotations)
            
            if pc.args.no_appearance_deformation: # without learnable deformation model on scales, opacity, and spherical harmonics
                scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
            else:
                scales_out, _rotations_out, opacity_out, shs_out = \
                    pose_conditioned_deform(means3D[None], means3D_deformed[None], scales[None], rotations[None], \
                                            opacity[None], shs[None], joints[None].float(), pc.appearance_deformation_model)
            
            #these are calculated using joint transformations
            means3D = means3D_deformed[0]
            rotations = rotations_deformed[0]
            
            #these are learned during lrs
            scales = scales_out[0]
            opacity = opacity_out[0]
            shs = shs_out[0]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    # viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    print(viewpoint_camera.world_view_transform, viewpoint_camera.full_proj_transform, render_features)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        render_features=render_features,
        render_gaussian_idx=render_gaussian_idx,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # means3D = pc.get_xyz 
    means2D = screenspace_points
    # opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    # scales = None
    # rotations = None
    cov3D_precomp = None
    # compute_cov3D_python = False
    # if compute_cov3D_python:
    #     cov3D_precomp = pc.get_covariance(scaling_modifier)
    # else:
    #     scales = pc.get_scaling
    #     rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    # shs = None
    colors_precomp = None
    # convert_SHs_python = False
    # if override_color is None:
    #     if convert_SHs_python:
    #         shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
    #         dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
    #         dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
    #         sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
    #         colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # (N, 3)
    #     else:
    #         shs = pc.get_features  # (N, 16 ,3)
    # else:
    #     colors_precomp = override_color
    
    # Get view-independent features (distill features) for each Gaussian for rendering.
    distill_feats = pc.get_distill_features
    
    # print(means3D, means2D, shs, colors_precomp, opacity, scales, rotations, cov3D_precomp, distill_feats)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_feat, rendered_depth, rendered_gaussian_idx, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp,
        distill_feats = distill_feats)
    
    # Default synthetic datasets settings
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,                # (N, 3)
    #     means2D = means2D,                # (N, 3)
    #     shs = shs,                        # (N, 16, 3)
    #     colors_precomp = colors_precomp,  # None
    #     opacities = opacity,              # 
    #     scales = scales,                  # (N, 3)
    #     rotations = rotations,            # (N, 4)
    #     cov3D_precomp = cov3D_precomp)    # None

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_feat": rendered_feat,
            "render_depth": rendered_depth,
            "render_gaussian_idx": rendered_gaussian_idx,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render(viewpoint_camera, 
           pc : GaussianModel, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           stage='pose_conditioned',
           render_features = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    assert stage in ['canonical', 'pose_conditioned']
    
    
    stage = 'pose_conditioned'

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )


    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

   
    joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    # print('joints', joints[0])
    if stage == 'pose_conditioned':
        if pc.args.k_plane: # using k-plane deformation model for deformation instead of linear blend skinning
            means3D, scales, rotations, opacity, shs = pc._deformation(means3D, scales, rotations, opacity, shs, times_sel=None, joints=joints)
            rotations = pc.rotation_activation(rotations)
        else: # use lbs for means+rotations, + MLP for appearance
            means3D_deformed, rotations_deformed = lrs(joints[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=True, lrs_model=pc.lrs_model, rotations=rotations)
            
            if pc.args.no_appearance_deformation: # without learnable deformation model on scales, opacity, and spherical harmonics
                scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
            else:
                scales_out, _rotations_out, opacity_out, shs_out = \
                    pose_conditioned_deform(means3D[None], means3D_deformed[None], scales[None], rotations[None], \
                                            opacity[None], shs[None], joints[None].float(), pc.appearance_deformation_model)
            
            #these are calculated using joint transformations
            means3D = means3D_deformed[0]
            rotations = rotations_deformed[0]
            
            #these are learned during lrs 
            scales = scales_out[0]
            opacity = opacity_out[0]
            shs = shs_out[0]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # viewmat = viewmat.requires_grad_(True)
    # print("viewmat shape", viewmat.shape, "supposed to be 4x4")
    # print("grad before", viewpoint_camera.v.grad)
    # breakpoint()

    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=int(viewpoint_camera.image_width),
        height=int(viewpoint_camera.image_height),
        packed=False,
        sh_degree=sh_degree,
    )

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass
    
    distill_features = None
    distill_features_3d = pc.get_distill_features
    background_features = torch.zeros(1, distill_features_3d.shape[-1], dtype=distill_features_3d.dtype, device=distill_features_3d.device)
    background_features[0, 3:] = 1
    # print(distill_features_3d.shape) 
    if render_features:
        distill_features, render_alphas, info1 = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=distill_features_3d,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=background_features,
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=None,
        )
        
    # print("render_features", distill_features)
    # print(distill_features.shape)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_feat": distill_features,}

def render_gradio(viewpoint_camera, 
           pc : GaussianModel, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           stage='pose_conditioned',
           render_features = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    assert stage in ['canonical', 'pose_conditioned']
    
    
    stage = 'pose_conditioned'

    # Set up rasterization configuration
    # tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    # tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    # focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    
    tanfovx = torch.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = torch.tan(viewpoint_camera.FoVy * 0.5)

    # Compute focal lengths
    focal_length_x = viewpoint_camera.image_width / (2 * tanfovx)
    focal_length_y = viewpoint_camera.image_height / (2 * tanfovy)
    
    
    # print(focal_length_x.requires_grad, 'inside render')
    
    means3D = pc.get_xyz
    opacity = pc.get_opacity
    
    focal_length_x = focal_length_x.to(means3D.device)
    focal_length_y = focal_length_y.to(means3D.device)
    # zero = torch.zeros(1, device=means3D.device, dtype=means3D.dtype)
    # one = torch.ones(1, device=means3D.device, dtype=means3D.dtype)
    # width_half = torch.tensor([viewpoint_camera.image_width / 2.0], device=means3D.device, dtype=means3D.dtype)
    # height_half = torch.tensor([viewpoint_camera.image_height / 2.0], device=means3D.device, dtype=means3D.dtype)

    # row1 = torch.cat([focal_length_x.unsqueeze(0), zero, width_half], dim=0)
    # row2 = torch.cat([zero, focal_length_y.unsqueeze(0), height_half], dim=0)
    # row3 = torch.cat([zero, zero, one], dim=0)

    # K = torch.stack([row1, row2, row3], dim=0)
    # K.retain_grad()

    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_camera.image_width / 2.0],
            [0, focal_length_y, viewpoint_camera.image_height / 2.0],
            [0, 0, 1],
        ],
    ).to(means3D.device)
    # print(K)
    # print(K.requires_grad)
    
    # K = torch.tensor(
    #     [
    #         [0, 0, viewpoint_camera.image_width / 2.0],
    #         [0, 0, viewpoint_camera.image_height / 2.0],
    #         [0, 0, 1],
    #     ],
    #     device="cuda",
    # )
    # K[0, 0] = focal_length_x
    # K[1, 1] = focal_length_y


    

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

   
    joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    # print('joints', joints[0])
    if stage == 'pose_conditioned':
        if pc.args.k_plane: # using k-plane deformation model for deformation instead of linear blend skinning
            means3D, scales, rotations, opacity, shs = pc._deformation(means3D, scales, rotations, opacity, shs, times_sel=None, joints=joints)
            rotations = pc.rotation_activation(rotations)
        else: # use lbs for means+rotations, + MLP for appearance
            means3D_deformed, rotations_deformed = lrs(joints[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=False, lrs_model=pc.lrs_model, rotations=rotations)
            
            if pc.args.no_appearance_deformation: # without learnable deformation model on scales, opacity, and spherical harmonics
                scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
            else:
                scales_out, _rotations_out, opacity_out, shs_out = \
                    pose_conditioned_deform(means3D[None], means3D_deformed[None], scales[None], rotations[None], \
                                            opacity[None], shs[None], joints[None].float(), pc.appearance_deformation_model)
            
            #these are calculated using joint transformations
            means3D = means3D_deformed[0]
            rotations = rotations_deformed[0]
            
            #these are learned during lrs
            scales = scales_out[0]
            opacity = opacity_out[0]
            shs = shs_out[0]
            color = shs[:, 0, :]
            # print(color.shape)
            pcd = np.concatenate([means3D.cpu().detach().numpy(), color.cpu().detach().numpy()], axis=1)

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    viewmat = viewpoint_camera.world_view_transform.transpose(0, 1).to(means3D.device)
    # print(K, viewmat, 'inside render')
    # viewmat = viewmat.requires_grad_(True)
    # print("viewmat shape", viewmat.shape, "supposed to be 4x4")
    # print("grad before", viewpoint_camera.v.grad)
    # breakpoint()

    # render_colors, render_alphas, info = rasterization(
    #     means=means3D,  # [N, 3]
    #     quats=rotations,  # [N, 4]
    #     scales=scales,  # [N, 3]
    #     opacities=opacity.squeeze(-1),  # [N,]
    #     colors=shs,
    #     viewmats=viewmat[None],  # [1, 4, 4]
    #     Ks=K[None],  # [1, 3, 3]
    #     backgrounds=bg_color[None],
    #     width=int(viewpoint_camera.image_width),
    #     height=int(viewpoint_camera.image_height),
    #     packed=False,
    #     sh_degree=sh_degree,
    #     render_mode='RGB+D'
    # )
    render_colors, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=shs,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=bg_color[None],
        width=torch.tensor(viewpoint_camera.image_width, device=means3D.device, dtype=means3D.dtype),
        height=torch.tensor(viewpoint_camera.image_height, device=means3D.device, dtype=means3D.dtype),
        packed=False,
        sh_degree=sh_degree,
        render_mode='RGB+D'
    )
    rendered_depth = render_colors[0, :, :, 3]
    render_colors = render_colors[:, :, :, :3]
    
    # loss = render_colors.sum()
    # loss.backward(retain_graph=True)

    # # Check gradients
    # print("K.grad:", K.grad)

    # [1, H, W, 3] -> [3, H, W]
    rendered_image = render_colors[0].permute(2, 0, 1)
    radii = info["radii"].squeeze(0) # [N,]
    try:
        info["means2d"].retain_grad() # [1, N, 2]
    except:
        pass
    
    distill_features = None
    distill_features_3d = pc.get_distill_features
    background_features = torch.zeros(1, distill_features_3d.shape[-1], dtype=distill_features_3d.dtype, device=distill_features_3d.device)
    # print(distill_features_3d.shape) 
    if render_features:
        distill_features, render_alphas, info1 = rasterization(
            means=means3D,  # [N, 3]
            quats=rotations,  # [N, 4]
            scales=scales,  # [N, 3]
            opacities=opacity.squeeze(-1),  # [N,]
            colors=distill_features_3d,
            viewmats=viewmat[None],  # [1, 4, 4]
            Ks=K[None],  # [1, 3, 3]
            backgrounds=background_features,
            width=int(viewpoint_camera.image_width),
            height=int(viewpoint_camera.image_height),
            packed=False,
            sh_degree=None,
        )
        
    # print("render_features", distill_features)
    # print(distill_features.shape)


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": info["means2d"],
            "visibility_filter" : radii > 0,
            "radii": radii,
            "render_feat": distill_features,
            "pcd": pcd,
            "depth": rendered_depth}
    

def point_cloud_to_image(world_coordinates, viewpoint_camera, K):

    means3D_homo = torch.cat([world_coordinates, torch.ones_like(world_coordinates[:, :1])], dim=1)
    means_clip = (viewpoint_camera.full_proj_transform.T @ means3D_homo.T).T
    depth = means_clip[:, 2]
    means_screen = means_clip / (means_clip[:, 3:] + 1e-6)
    x_normalized = means_screen[:, 0] * 0.5 + 0.5
    y_normalized = means_screen[:, 1] * 0.5 + 0.5

    x_pixel = x_normalized * int(viewpoint_camera.image_width)
    y_pixel = y_normalized * int(viewpoint_camera.image_height)

    coords = torch.stack([x_pixel, y_pixel], axis=1)

    return coords, depth

def project_3d_to_2d(points_3d, T_wc, K):
    """
    Project 3D world points to 2D image coordinates using homogeneous coordinates in PyTorch.
    
    Args:
        points_3d (torch.Tensor): [N, 3] tensor of 3D points in world coordinates.
        T_wc (torch.Tensor): [4, 4] world-to-camera extrinsic transformation matrix.
        K (torch.Tensor): [3, 3] camera intrinsic matrix.
    
    Returns:
        torch.Tensor: [N, 2] tensor of 2D image coordinates.
    """
    # Convert to homogeneous coordinates
    N = points_3d.size(0)
    ones = torch.ones(N, 1, device=points_3d.device, dtype=points_3d.dtype)
    points_world_hom = torch.cat([points_3d, ones], dim=1)  # [N, 4]
    
    # Compute projection matrix
    P = K @ T_wc[:3, :4]  # [3, 4]
    
    # Project to image plane
    points_2d_hom = P @ points_world_hom.t()  # [3, N]
    
    # Normalize to 2D coordinates
    points_2d = points_2d_hom[:2, :] / points_2d_hom[2, :].unsqueeze(0)  # [2, N]
    points_2d = points_2d.t()  # [N, 2]
    
    return points_2d

def render_flow(viewpoint_cameras,  # [cam_curr, cam_prev]
           pc : GaussianModel, 
           bg_color : torch.Tensor, 
           scaling_modifier = 1.0, 
           override_color = None, 
           stage='pose_conditioned',
           render_features = True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    assert stage in ['canonical', 'pose_conditioned']
    
    
    stage = 'pose_conditioned'

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cameras[1].FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cameras[1].FoVy * 0.5)

    focal_length_x = viewpoint_cameras[1].image_width / (2 * tanfovx)
    focal_length_y = viewpoint_cameras[1].image_height / (2 * tanfovy)
    K = torch.tensor(
        [
            [focal_length_x, 0, viewpoint_cameras[1].image_width / 2.0],
            [0, focal_length_y, viewpoint_cameras[1].image_height / 2.0],
            [0, 0, 1],
        ],
        device="cuda",
    )


    means3D = pc.get_xyz
    opacity = pc.get_opacity

    scales = pc.get_scaling * scaling_modifier
    rotations = pc.get_rotation

    if override_color is not None:
        shs = override_color # [N, 3]
        sh_degree = None
    else:
        shs = pc.get_features # [N, K, 3]
        sh_degree = pc.active_sh_degree

   
    # joints = viewpoint_camera.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    # print('joints', joints[0])
    assert stage == 'pose_conditioned'
    viewpoint_cam1 = viewpoint_cameras[0]
    viewpoint_cam2 = viewpoint_cameras[1]
    joints1 = viewpoint_cam1.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    joints2 = viewpoint_cam2.joint_pose.to(means3D.device).repeat(means3D.shape[0],1)
    
    means3D_deformed1, rotations_deformed1 = lrs(joints1[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=False, lrs_model=pc.lrs_model, rotations=rotations)
    means3D_deformed2, rotations_deformed2 = lrs(joints2[0][None].float(), means3D[None], pc._lrs, pc.chain, pose_normalized=False, lrs_model=pc.lrs_model, rotations=rotations)
    scales_out, _rotations_out, opacity_out, shs_out = scales[None], rotations[None], opacity[None], shs[None]
        
    #these are calculated using joint transformations
    means3D1 = means3D_deformed1[0]
    rotations1 = rotations_deformed1[0]
    means3D2 = means3D_deformed2[0]
    rotations2 = rotations_deformed2[0]
    means3D = means3D_deformed2[0]
    rotations = rotations_deformed2[0]
    viewmat = viewpoint_cameras[1].world_view_transform.transpose(0, 1).to(means3D.device)
    gaussian_2d_pos_curr, _ = point_cloud_to_image(means3D1, viewpoint_cameras[0], K)
    gaussian_2d_pos_prev, _ = point_cloud_to_image(means3D2, viewpoint_cameras[1], K)
    # gaussian_2d_pos_curr = project_3d_to_2d(means3D1, viewmat, K)
    # gaussian_2d_pos_prev = project_3d_to_2d(means3D2, viewmat, K)
    
    flow_2d = gaussian_2d_pos_curr - gaussian_2d_pos_prev
    # print(flow_2d.shape, 'flow2d')
    flow_padded = torch.cat([flow_2d, torch.zeros_like(flow_2d[:, 1:])], dim=1)
        
    #these are learned during lrs 
    scales = scales_out[0]
    opacity = opacity_out[0]
    shs = shs_out[0]

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    # breakpoint()
    
    # viewmat = viewmat.requires_grad_(True)
    # print("viewmat shape", viewmat.shape, "supposed to be 4x4")
    # print("grad before", viewpoint_camera.v.grad)
    # breakpoint()

    rendered_flow_image, render_alphas, info = rasterization(
        means=means3D,  # [N, 3]
        quats=rotations,  # [N, 4]
        scales=scales,  # [N, 3]
        opacities=opacity.squeeze(-1),  # [N,]
        colors=flow_2d,
        viewmats=viewmat[None],  # [1, 4, 4]
        Ks=K[None],  # [1, 3, 3]
        backgrounds=torch.zeros((1, 2)).cuda(),
        width=int(viewpoint_cameras[1].image_width),
        height=int(viewpoint_cameras[1].image_height),
        packed=False,
        sh_degree=None,
    )


    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"flow": rendered_flow_image,
            "viewspace_points": info["means2d"],
            "gaussian_2d_pos_curr": gaussian_2d_pos_curr}