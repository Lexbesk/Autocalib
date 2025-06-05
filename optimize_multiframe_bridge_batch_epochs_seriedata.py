import sys
import tempfile
import os
import cv2
import time
os.environ['MUJOCO_GL'] = 'egl'
from torch.optim.lr_scheduler import StepLR
from scipy.spatial.transform import Rotation

tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir
if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')
    
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import numpy as np
import mujoco
from utils_loc.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from utils_loc.pk_utils import build_chain_from_mjcf_path
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import math
from transformations import quaternion_from_matrix
from modern_robotics.core import JacobianSpace, Adjoint, MatrixLog6, se3ToVec, TransInv, FKinSpace
from tqdm import tqdm

from video_api import initialize_gaussians
from gaussian_renderer import render, render_gradio, render_flow
from scene.cameras import Camera_Pose, Camera
from utils_loc.flow_utils import run_flow_on_images, run_tracker_on_images
import torch
import torch.nn.functional as F
from PIL import Image

from scipy.spatial.transform import Rotation as R
from scene import RobotScene, GaussianModel, feat_decoder, skip_feat_decoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from compute_image_dino_feature import resize_image, interpolate_to_patch_size, resize_tensor
import torchvision.transforms as T
from utils_loc.loss_utils import l1_loss, ssim, cosine_loss
import pickle
from optimize_multiframe_droid_batch import flow_to_color, iou_loss
from utils_loc.generate_grid_campose import generate_camera_poses, perturb_extrinsic


def visualize_features(features, path):
    """
    Visualize a DINOv2 feature map by reducing its dimensionality to 3 using PCA.

    Args:
        features (np.ndarray): Feature map of shape (H, W, D), where D is the feature dimension (>700).

    Notes:
        - If your feature map is a PyTorch tensor, convert it to NumPy with features.cpu().numpy().
        - If it's a flat array of shape (N, D), reshape it to (H, W, D) first, where N = H * W.
    """
    # Get the dimensions of the feature map
    Hf, Wf, D = features.shape
    print(f"Feature map shape: {H}x{W}x{D}")

    # Reshape to (H*W, D) for PCA
    features_reshaped = features.reshape(Hf * Wf, D)

    # Apply PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    features_pca = pca.fit_transform(features_reshaped)
    print(f"Explained variance ratio of 3 components: {pca.explained_variance_ratio_}")

    # Reshape back to (H, W, 3) for visualization
    features_pca_image = features_pca.reshape(Hf, Wf, 3)

    # Normalize each channel to [0,1] for RGB display
    for i in range(3):
        channel = features_pca_image[:, :, i]
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:  # Avoid division by zero
            features_pca_image[:, :, i] = (channel - min_val) / (max_val - min_val)
        else:
            features_pca_image[:, :, i] = 0.5  # Set to mid-range if constant

    # Display the visualization
    plt.figure(figsize=(8, 8))
    plt.imshow(features_pca_image)
    plt.axis('off')
    plt.title("DINOv2 Feature Visualization (PCA to RGB)")
    plt.savefig(path, bbox_inches='tight')



def optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    mujoco_feats = []
    mujoco_direct_feats = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if render_feat or use_direct_feat:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    os.makedirs('inspection_features', exist_ok=True)
    os.makedirs('inspection_features_1', exist_ok=True)
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        if render_feat:
            dino_image = mujoco_image
            # print(dino_image.shape, 'dino_image')
            dino_image = Image.fromarray(dino_image)
            dino_image.save(f'inspection_features/dino_image_{c_idx}.png')
            dino_image = resize_image(dino_image, 800)
            dino_image = dino_transform(dino_image)[:3].unsqueeze(0)
            dino_image, target_H, target_W = interpolate_to_patch_size(dino_image, dinov2.patch_size)
            dino_image = dino_image.cuda()
            with torch.no_grad():
                features = dinov2.forward_features(dino_image)["x_norm_patchtokens"][0]
            features = features.cpu().numpy()
            # print(features.shape, 'features')
            features_hwc = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            features_hwc = torch.from_numpy(features_hwc).float().cuda()
            # print(features_hwc.shape, 'features_hwc')
            mujoco_feats.append(features_hwc)
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
        if have_depth:
            mujoco_depth = depth_list[c_idx]
            mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
            mujoco_depth_tensor = mujoco_depth_tensor
            mujoco_depths.append(mujoco_depth_tensor)
        # Only perturb parameters that are being optimized
        # print("gaussian_params", gaussian_params)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)

    optimizers = []
    schedulers = []
    if optimize_camera:
        optimizer = torch.optim.Adam([param for camera in all_cameras for param in camera.parameters()], lr=camera_lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    if optimize_joints:
        optimizers.append(torch.optim.Adam(all_joint_poses, lr=joints_lr))

    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        for optimizer in optimizers:
            optimizer.zero_grad()

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_feats = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features=render_feat)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features=False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
                    
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
            if render_feat:
                rendered_feat = output['render_feat'].permute(0, 3, 1, 2)
                dino_feat = my_feat_decoder(rendered_feat).permute(0, 2, 3, 1).squeeze()
                gaussian_feats.append(dino_feat)
            if step == 0 and i in [0, 10, 19]:
                # print(f'-----------------------------------saving pcd from {i}')
                os.makedirs("pcd", exist_ok=True)
                np.save(f"pcd/means3D_{i}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            os.makedirs('tmp', exist_ok=True)
            mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        if render_feat:
            gaussian_feats = torch.stack(gaussian_feats)
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        if render_feat:
            mujoco_feats_target = torch.stack(mujoco_feats)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        # print(gaussian_flows.shape, 'gaussian flows')
        # print(step, 'step')
        have_motion = True
        if tracking_loss and step == 0:
            if use_sam_mask:
                mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            if have_depth:
                mujoco_depths_target = torch.stack(mujoco_depths)
            # print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.4)
            print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            motion_idx = (motion_mask.sum(dim=(1, 2, 3)) >= 10000) & (motion_mask.sum(dim=(1, 2, 3)) < 500000)
            # print()
            motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # print(motion_idx, motion_idx.shape)
            if motion_idx.sum() <= 1:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid images < 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                have_motion = False
                torch.cuda.empty_cache()
                break
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            if have_depth:
                depth_mask = (mujoco_depths_target[:-1] > 0.5)
                depth_near_mask1 = (mujoco_depths_target < 0.7)
                depth_near_mask = (mujoco_depths_target[:-1] < 0.7)
                depth_near_mask = depth_near_mask.unsqueeze(1).float() # [T-1, 1, H, W]
                motion_depth_near_mask = motion_mask * depth_near_mask
            # print(mujoco_depths_target)
            # print(depth_mask)
                depth_mask = depth_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            
            # print(depth_mask.shape, 'depth mask shape')
            # print(motion_mask)
            # print(motion_mask.shape, 'motion mask shape')
            
            
            
        if tracking_loss and step % 50 == 1 and epoch == 0:
            for i in range(1, min(30, mujoco_images_target.shape[0])):
                fig, axes = plt.subplots(3, 3, figsize=(15, 15))
                
                flow_2d_np_render = gaussian_flows[i-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                step_grid = 10
                new_image_render = gaussian_tensors[i].permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np_render = flow_to_color(flow_2d_np_render)
                H_f, W_f = flow_2d_np_render.shape[:2]
                x_render, y_render = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx_render = flow_2d_np_render[y_render, x_render, 0]
                dy_render = flow_2d_np_render[y_render, x_render, 1]
                
                flow_2d_mujoco = (fwd_flows * fwd_valids * motion_mask)[i-1].permute(1, 2, 0).detach().cpu().numpy()
                new_image_np = (mujoco_images_target[i] / 255).permute(1, 2, 0).detach().cpu().numpy()
                flow_color_mujoco = flow_to_color(flow_2d_mujoco)
                step_grid = 10
                H_f, W_f = flow_2d_mujoco.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_mujoco[y, x, 0]
                dy = flow_2d_mujoco[y, x, 1]
                
                mask_i = motion_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
                if have_depth:
                    depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                    depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                    
                    motion_depth_near_mask_i = motion_depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                # print(flow_valid_mask_i.min(), flow_valid_mask_i.max(), 'flow valid mask min max')
                

                axes[0, 0].imshow(new_image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(flow_color_mujoco)
                axes[0, 1].set_title('Flow Color Map')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(new_image_np)
                axes[0, 2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[0, 2].set_title('Image with Flow Arrows')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(new_image_render)
                axes[1, 0].set_title('Rendered Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(flow_color_np_render)
                axes[1, 1].set_title('Rendered Flow Color Map')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(new_image_render)
                axes[1, 2].quiver(x, y, dx_render, dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[1, 2].set_title('Rendered Flow Arrows')
                axes[1, 2].axis('off')

                axes[2, 0].imshow(new_image_np)
                axes[2, 0].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 0].set_title('Motion Mask')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(new_image_np)
                axes[2, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 1].set_title('Depth Mask')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(new_image_np)
                axes[2, 2].quiver(x, y, dx - dx_render, dy - dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2, 2].set_title('Image with Flow Arrows')
                axes[2, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features_1/super_flow_visualization{i}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
    
        image = gaussian_tensors
        mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        if use_direct_feat:
            resized_image = resize_tensor(image, longest_edge=800)
            resized_mujo_image = resize_tensor(mujoco_images_target1, longest_edge=800)
            
            image_final = normalize(resized_image)
            mujo_final = normalize(resized_mujo_image)
            dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
            dino_mojo, target_H, target_W = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
            features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
            mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
            # features = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            
            resized_mask = F.interpolate(gaussian_masks.unsqueeze(1).float(), size=(target_H // dinov2.patch_size, target_W // dinov2.patch_size), mode='nearest')
            # print(resized_mask.shape, 'resize mask')
            resized_mask = resized_mask.reshape(features.shape[0], -1)
            # print(resized_mask.shape, 'resize mask')
            # print(features.shape, 'features shape')
            features = features * resized_mask.squeeze().unsqueeze(-1)
            loss_direct_feat = cosine_loss(features, mujo_features)
            # print(loss_direct_feat.item(), 'direct feature loss')
            features_np = features[0].detach().cpu().numpy()
            mujo_features_np = mujo_features[0].detach().cpu().numpy()
            features_hwc = features_np.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            mujo_features_hwc = mujo_features_np.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            # print(features_hwc.shape, 'features_np shape')
            # print(mujo_features_hwc.shape, 'mujoco_features_np shape')
            features_final = np.concatenate([features_hwc, mujo_features_hwc], axis=0)
            visualize_features(features_final, f'inspection_features/features_{0}.png')

        
        if tracking_loss:
            flow_ratio = gaussian_flows.max() / fwd_flows.max()
            print(flow_ratio, 'flow ratio')
            # if flow_ratio > 4:
            #     # adjust intrinsics to a smaller focal length
            #     intrinsics[0, 0] -= 5
            #     intrinsics[1, 1] -= 5
            # elif flow_ratio < 0.25:
            #     # adjust intrinsics to a larger focal length
            #     intrinsics[0, 0] += 5
            #     intrinsics[1, 1] += 5
            flow_loss = (gaussian_flows - fwd_flows).abs()
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
            # if not have_motion:
            #     flow_loss *= 0
        
        if vis_features and step % 5 == 0:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features_1/images_{i}.png')
                
                image_render = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np_render = image_render.to(torch.uint8)
                image_np_render = image_np_render.detach().cpu().numpy()
                image_np_render = Image.fromarray(image_np_render)
                image_np_final = Image.fromarray(np.concatenate([image_np_render, image_np], axis=1))
                image_np_final.save(f'inspection_features_1/image_render_{i}.png')
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format
        if reg_mask:
            gaussian_mask_sum = gaussian_masks.sum()
            gaussian_mask_sum = gaussian_mask_sum / gaussian_masks.shape[0]
            reg_mask_loss = (10000 - gaussian_mask_sum)
            print(reg_mask_loss.item(), 'reg mask loss')
        if use_sam_mask:
            ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
            ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()
            ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        

        # Step 4: Save masks as images
        os.makedirs('masks_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
            if use_sam_mask:
                gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
                cv2.imwrite(gt_filename, ground_truth_masks_np[i])
        
        if have_depth:
            gaussian_depths_np = gaussian_depths.detach().cpu().numpy()
            mujoco_depths_np = mujoco_depths_target.detach().cpu().numpy()
            gaussian_depths_np *= 100
            mujoco_depths_np *= 100
            os.makedirs('depths_inspection', exist_ok=True)
            for i in range(gaussian_masks_np.shape[0]):
                # Save Gaussian mask
                gaussian_filename = f"depths_inspection/gaussian_depth_{i}.png"
                cv2.imwrite(gaussian_filename, gaussian_depths_np[i])
                # print(f"Saved {gaussian_filename}")
                # Save Ground Truth mask
                gt_filename = f"depths_inspection/ground_truth_depth_{i}.png"
                cv2.imwrite(gt_filename, mujoco_depths_np[i])
                # print(f"Saved {gt_filename}")
                gt_filename = f"depths_inspection/compare_depth_{i}.png"
                cv2.imwrite(gt_filename, np.abs(mujoco_depths_np[i] - gaussian_depths_np[i]))
        
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        if use_sam_mask:
            IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
            # print(IoU_loss, 'IoU_loss')
            
        
        if have_depth:
            IoU_loss_motion = iou_loss(gaussian_masks[:-1], motion_depth_near_mask)
            depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
            print(depth_loss.shape, 'depth_loss shape', depth_near_mask1.shape)
            depth_diff = (depth_loss * depth_near_mask1).mean()

        # total_loss = l2_diff.sum()
        # total_loss = loss_feat + IoU_loss
        # total_loss = l2_diff.sum() + loss_feat + depth_diff * 1 + IoU_loss * 10
        flow_loss = flow_loss * 1000
        # flow_loss_motion = flow_loss_motion * 10
        if have_depth:
            depth_diff = depth_diff * 0.1
            IoU_loss_motion = IoU_loss_motion * 0.01
        # total_loss = flow_loss + flow_loss_motion
        # total_loss = depth_diff * 0.01 + flow_loss
        # if flow_loss == 0:
        #     print('Flow loss is zero, skipping optimization step.')
        #     continue
        total_loss = flow_loss + l2_diff * 0.1
        # total_loss = 0
        if use_direct_feat:
            loss_direct_feat = loss_direct_feat * 1
            total_loss += loss_direct_feat
        print('flow_loss :', flow_loss.item())
        if use_direct_feat:
            print('loss_feat:', loss_direct_feat.item())
        print('rgb loss:', l2_diff.item())
        if have_depth:
            print('depth_diff:', depth_diff.item(),  'IoU_loss_motion :', IoU_loss_motion.item())
        # print('total_loss: ', total_loss.item())
        print('epoch:', epoch, Optimize_list)
        
        # if IoU_loss > 0.85: 
        #     total_loss = l2_diff * 3 + depth_diff * 0.1 + IoU_loss * 10
        # else:
        #     total_loss = l2_diff + depth_diff * 1 + IoU_loss * 10
        # total_loss = 10 * depth_diff + 0.1 * IoU_loss
        total_loss.backward()

        for optimizer in optimizers:
            optimizer.step()
        for scheduler in schedulers:
            scheduler.step()    

    # print('Camera results robot to world transformation:')
    # for i in range(len(all_cameras)):
    #     print(all_cameras[i].robot_to_world())
    #     print(all_cameras[i].get_camera_pose().tolist(), 'camera pose final')
    # print(best_camera.robot_to_world())
    valid_masks = False
    # if flow_loss < 0.5: # TODO: use other metrics
    valid_masks = True
    # yield (best_image, *rounded_final_params)
    # print(all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy())
    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), True
    

def compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if render_feat or use_direct_feat_compute:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # s, b, l, g
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    os.makedirs('inspection_features_1', exist_ok=True)
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
        if have_depth:
            mujoco_depth = depth_list[c_idx]
            mujoco_depth_tensor = torch.from_numpy(mujoco_depth).float().cuda()
            mujoco_depth_tensor = mujoco_depth_tensor
            mujoco_depths.append(mujoco_depth_tensor)
        # Only perturb parameters that are being optimized
        # print("gaussian_params", gaussian_params)
        
        joint_angles = torch.tensor(joints[c_idx])
        intrinsic_rh = intrinsics
        fx = intrinsic_rh[0, 0]
        fy = intrinsic_rh[1, 1]
        fovx = 2 * np.arctan(W / (2 * fx))
        fovy = 2 * np.arctan(H / (2 * fy))
        camera_extrinsic_matrix = extrinsics
        
        camera = Camera_Pose(torch.tensor(camera_extrinsic_matrix).clone().detach().float().cuda(), fovx, fovy,\
                                W, H, joint_pose=joint_angles, zero_init=True).cuda()
        if first_camera is None:
            first_camera = camera  # First camera keeps its initialized parameters
        else:
            # Replace subsequent cameras' parameters with first_camera's
            for name, param in first_camera.named_parameters():
                if name in camera._parameters:
                    camera._parameters[name] = param
        all_cameras.append(camera)
        all_joint_poses.append(joint_angles)


    for step in tqdm(range(optimization_steps)):
        all_losses = []
        all_images = []

        gaussian_tensors = []
        gaussian_masks = []
        gaussian_depths = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features=False)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features=False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
                    
            new_image = output['render']
            new_depth = output['depth']
            new_pcd = output['pcd']
            
            if step == 0:
                # print(f'-----------------------------------saving pcd from {i}')
                os.makedirs("pcd", exist_ok=True)
                np.save(f"pcd/means3D_{i}.npy", new_pcd)
            i += 1
            new_depth = torch.nan_to_num(new_depth)
            gaussian_tensors.append(new_image)
            gaussian_depths.append(new_depth)
            # mask_tensor = (new_image > 0.01).any(dim=0).float() # no grad
            thresholded = torch.sigmoid((new_image - 0.01) * 500)
            mask_tensor = thresholded.max(dim=0)[0]
            mask_to_save = (mask_tensor * 255).byte().cpu().numpy()
            mask_image = Image.fromarray(mask_to_save, mode='L')  # 'L' for grayscale
            os.makedirs('tmp', exist_ok=True)
            mask_image.save('tmp/extracted_mask.png')
            gaussian_masks.append(mask_tensor)
        # print(i, 'final')
            
        
        gaussian_tensors = torch.stack(gaussian_tensors)
        gaussian_masks = torch.stack(gaussian_masks)
        gaussian_depths = torch.stack(gaussian_depths)
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        # print(gaussian_flows.shape, 'gaussian flows')
        # print(step, 'step')
        if tracking_loss and step == 0:
            if use_sam_mask:
                mujoco_mask_target = torch.stack(mujoco_masks)
            mujoco_images_target = torch.stack(mujoco_images)
            if have_depth:
                mujoco_depths_target = torch.stack(mujoco_depths)
            # print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.4)
            # print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            motion_idx = motion_mask.sum(dim=(1, 2, 3)) > 10000
            motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            # print(motion_idx, motion_idx.shape)
            # if motion_idx.sum() <= 0:
            #     print('valid images < 5')
            #     torch.cuda.empty_cache()
            #     break
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            if have_depth:
                depth_mask = (mujoco_depths_target[:-1] > 0.5)
                depth_near_mask1 = (mujoco_depths_target < 0.7)
                depth_near_mask = (mujoco_depths_target[:-1] < 0.7)
                depth_near_mask = depth_near_mask.unsqueeze(1).float() # [T-1, 1, H, W]
                motion_depth_near_mask = motion_mask * depth_near_mask
            # print(mujoco_depths_target)
            # print(depth_mask)
                depth_mask = depth_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            
            # print(depth_mask.shape, 'depth mask shape')
            # print(motion_mask)
            # print(motion_mask.shape, 'motion mask shape')
            
            
            
        if tracking_loss and step % 19 == 1 and epoch == 0 and False:
            for i in range(1, min(2, mujoco_images_target.shape[0])):
                fig, axes = plt.subplots(4, 3, figsize=(15, 20))
                
                flow_2d_np_render = gaussian_flows[i-1].squeeze(0).permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 2]
                step_grid = 10
                new_image_render = gaussian_tensors[i].permute(1, 2, 0).detach().cpu().numpy()
                flow_color_np_render = flow_to_color(flow_2d_np_render)
                H_f, W_f = flow_2d_np_render.shape[:2]
                x_render, y_render = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx_render = flow_2d_np_render[y_render, x_render, 0]
                dy_render = flow_2d_np_render[y_render, x_render, 1]
                
                flow_2d_mujoco = fwd_flows[i-1].permute(1, 2, 0).detach().cpu().numpy()
                new_image_np = (mujoco_images_target[i] / 255).permute(1, 2, 0).detach().cpu().numpy()
                flow_color_mujoco = flow_to_color(flow_2d_mujoco)
                step_grid = 10
                H_f, W_f = flow_2d_mujoco.shape[:2]
                x, y = np.meshgrid(np.arange(0, W_f, step_grid), np.arange(0, H_f, step_grid))
                dx = flow_2d_mujoco[y, x, 0]
                dy = flow_2d_mujoco[y, x, 1]
                
                mask_i = motion_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
                if have_depth:
                    depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                    depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                    
                    motion_depth_near_mask_i = motion_depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                # print(flow_valid_mask_i.min(), flow_valid_mask_i.max(), 'flow valid mask min max')
                

                axes[0, 0].imshow(new_image_np)
                axes[0, 0].set_title('Original Image')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(flow_color_mujoco)
                axes[0, 1].set_title('Flow Color Map')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(new_image_np)
                axes[0, 2].quiver(x, y, dx, dy, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[0, 2].set_title('Image with Flow Arrows')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(new_image_render)
                axes[1, 0].set_title('Rendered Image')
                axes[1, 0].axis('off')

                axes[1, 1].imshow(flow_color_np_render)
                axes[1, 1].set_title('Rendered Flow Color Map')
                axes[1, 1].axis('off')

                axes[1, 2].imshow(new_image_render)
                axes[1, 2].quiver(x, y, dx_render, dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[1, 2].set_title('Rendered Flow Arrows')
                axes[1, 2].axis('off')

                axes[2, 0].imshow(new_image_np)
                axes[2, 0].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 0].set_title('Motion Mask')
                axes[2, 0].axis('off')

                axes[2, 1].imshow(new_image_np)
                axes[2, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[2, 1].set_title('Depth Mask')
                axes[2, 1].axis('off')

                axes[2, 2].imshow(new_image_np)
                axes[2, 2].quiver(x, y, dx - dx_render, dy - dy_render, color='red', angles='xy', scale_units='xy', scale=1, width=0.001)
                axes[2, 2].set_title('Image with Flow Arrows')
                axes[2, 2].axis('off')
                
                axes[3, 0].imshow(new_image_np)
                axes[3, 0].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 0].set_title('Depth Mask')
                axes[3, 0].axis('off')

                axes[3, 1].imshow(new_image_np)
                axes[3, 1].imshow(mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 1].set_title('Valid Mask')
                axes[3, 1].axis('off')

                axes[3, 2].imshow(new_image_np)
                axes[3, 2].imshow(flow_valid_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 2].set_title('Flow valid mask')
                axes[3, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features/super_flow_visualization{i}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
                
        image = gaussian_tensors
        mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        if use_direct_feat_compute:
            resized_image = resize_tensor(image, longest_edge=800)
            resized_mujo_image = resize_tensor(mujoco_images_target1, longest_edge=800)
            
            image_final = normalize(resized_image)
            mujo_final = normalize(resized_mujo_image)
            dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
            dino_mojo, target_H, target_W = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
            features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
            mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
            # features = features.reshape((target_H // dinov2.patch_size, target_W // dinov2.patch_size, -1))
            
            resized_mask = F.interpolate(gaussian_masks.unsqueeze(1).float(), size=(target_H // dinov2.patch_size, target_W // dinov2.patch_size), mode='nearest')
            # print(resized_mask.shape, 'resize mask')
            resized_mask = resized_mask.reshape(features.shape[0], -1)
            # print(resized_mask.shape, 'resize mask')
            # print(features.shape, 'features shape')
            features = features * resized_mask.squeeze().unsqueeze(-1)
            loss_direct_feat = cosine_loss(features, mujo_features)
            

        # if use_direct_feat_compute:
        #     print(gaussian_tensors.shape, 'gaussian tensors shape')
        #     resized_image = resize_tensor(gaussian_tensors, longest_edge=800)
        #     resized_mujo_image = resize_tensor(mujoco_images_target / 255, longest_edge=800)
            
        #     image_final = normalize(resized_image)
        #     mujo_final = normalize(resized_mujo_image)
        #     print(dinov2.patch_size, 'dino patch size')
        #     dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
        #     dino_mojo, _, _ = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
        #     features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
        #     mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
        #     p = dinov2.patch_size  # Typically 14 for DINOv2
        #     grid_h = target_H // p  # Number of patches along height
        #     grid_w = target_W // p  # Number of patches along width
            
        #     # Reshape to (batch_size, feature_dim, grid_h, grid_w) for pooling
        #     feature_dim = features.shape[-1]  # e.g., 768
        #     features_reshaped = features.view(1, grid_h, grid_w, feature_dim).permute(0, 3, 1, 2)
        #     mujo_features_reshaped = mujo_features.view(1, grid_h, grid_w, feature_dim).permute(0, 3, 1, 2)
            
        #     # Apply 2x2 average pooling to reduce spatial dimensions by half
        #     pool = torch.nn.AvgPool2d(kernel_size=4, stride=4)
        #     features_pooled = pool(features_reshaped)
        #     mujo_features_pooled = pool(mujo_features_reshaped)
            
        #     # Reshape back to (batch_size, new_num_patches, feature_dim)
        #     new_grid_h = features_pooled.shape[2]  # Reduced height
        #     new_grid_w = features_pooled.shape[3]  # Reduced width
        #     features_pooled = features_pooled.permute(0, 2, 3, 1).view(1, new_grid_h * new_grid_w, feature_dim)
        #     mujo_features_pooled = mujo_features_pooled.permute(0, 2, 3, 1).view(1, new_grid_h * new_grid_w, feature_dim)
            
        #     # Compute the loss with downsampled features
        #     loss_direct_feat = cosine_loss(features_pooled, mujo_features_pooled)
        #     # loss_direct_feat = cosine_loss(features, mujo_features)
        #     print(loss_direct_feat.item(), 'direct feature loss')
            
        
        if tracking_loss:
            # compare the magnitude of flows, compute the ratio between the max flow
            # flow_ratio = gaussian_flows.max() / fwd_flows.max()
            # print(flow_ratio, 'flow ratio')
            # if flow_ratio > 4:
            #     # adjust intrinsics to a smaller focal length
            #     intrinsics[0, 0] -= 5
            #     intrinsics[1, 1] -= 5
            # elif flow_ratio < 0.25:
            #     # adjust intrinsics to a larger focal length
            #     intrinsics[0, 0] += 5
            #     intrinsics[1, 1] += 5
            flow_loss = (gaussian_flows - fwd_flows).abs()
            
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
        
        if vis_features:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features_1/images_{i}.png')
                
                image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features_1/image_render_{i}.png')
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format
        if use_sam_mask:
            ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
            ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()
            ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        

        # Step 4: Save masks as images
        os.makedirs('masks_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_masks_np[i])
            if use_sam_mask:
                gt_filename = f"masks_inspection/ground_truth_mask_{i}.png"
                cv2.imwrite(gt_filename, ground_truth_masks_np[i])
        
        if have_depth:
            gaussian_depths_np = gaussian_depths.detach().cpu().numpy()
            mujoco_depths_np = mujoco_depths_target.detach().cpu().numpy()
            gaussian_depths_np *= 100
            mujoco_depths_np *= 100
            os.makedirs('depths_inspection', exist_ok=True)
            for i in range(gaussian_masks_np.shape[0]):
                # Save Gaussian mask
                gaussian_filename = f"depths_inspection/gaussian_depth_{i}.png"
                cv2.imwrite(gaussian_filename, gaussian_depths_np[i])
                # print(f"Saved {gaussian_filename}")
                # Save Ground Truth mask
                gt_filename = f"depths_inspection/ground_truth_depth_{i}.png"
                cv2.imwrite(gt_filename, mujoco_depths_np[i])
                # print(f"Saved {gt_filename}")
                gt_filename = f"depths_inspection/compare_depth_{i}.png"
                cv2.imwrite(gt_filename, np.abs(mujoco_depths_np[i] - gaussian_depths_np[i]))
        
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        
        # print(loss_feat, 'loss_feat')
        if use_sam_mask:
            IoU_loss = iou_loss(gaussian_masks, mujoco_mask_target)
            print(IoU_loss, 'IoU_loss')
            

        flow_loss = flow_loss * 1000 
        if flow_loss == 0:
            return None, float('inf')
        if use_direct_feat_compute:
            print('flow_loss :', flow_loss.item(), 'loss_feat:', loss_direct_feat.item(), 'rgb loss:', l2_diff.item())
        else:
            print('flow_loss :', flow_loss.item(), 'rgb loss:', l2_diff.item())
        

    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), loss_direct_feat.item()
    



def prepare_flow(images, device, data_file):
    mujoco_images = []
    for c_idx in range(len(images)):
        mujoco_image = images[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
    mujoco_images_target = torch.stack(mujoco_images)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
    # Run cotracker
    fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target) # T-1 2 H W, T-1 H W
    fwd_valids = fwd_valids.unsqueeze(1)
    # Prepare the data by moving tensors to CPU and storing in a dictionary
    data_to_save = {
        'fwd_flows': fwd_flows.detach().cpu(),
        'fwd_valids': fwd_valids.detach().cpu()
    }
    # Save the dictionary to the file
    torch.save(data_to_save, data_file)

def grid_search():
    global extrinsics
    cam_poses = generate_camera_poses()
    best_pose = None
    best_score = float('inf')
    for pose in cam_poses:
        extrinsics = euler_to_extrinsic(pose)
        optimize_camera = True
        optimize_joints = False
        camera_lr = 0.0
        joints_lr = 0.0
        optimization_steps = 1
        params, flow_loss = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                    fwd_flows, fwd_valids)
        
        


def euler_to_extrinsic(pose_6d, euler_order='xyz'):
    """
    Convert a 6D camera pose (translation + Euler angles) to a 4x4 extrinsic matrix.

    Args:
        pose_6d (np.ndarray): 6-element array with [x, y, z, alpha, beta, gamma].
        euler_order (str): Order of Euler angles (e.g., 'xyz', 'zyx'). Default is 'xyz'.

    Returns:
        np.ndarray: 4x4 extrinsic camera matrix.
    """
    # Extract translation vector
    t = pose_6d[:3]

    # Extract Euler angles
    euler_angles = pose_6d[3:]

    # Convert Euler angles to rotation matrix
    R = Rotation.from_euler(euler_order, euler_angles).as_matrix()

    # Initialize 4x4 extrinsic matrix
    extrinsic = np.eye(4)

    # Assign rotation and translation
    extrinsic[:3, :3] = R
    extrinsic[:3, 3] = t

    extrinsic = np.linalg.inv(extrinsic)

    return extrinsic

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.
    
    Args:
        roll (float): Roll angle in radians.
        pitch (float): Pitch angle in radians.
        yaw (float): Yaw angle in radians.
    
    Returns:
        tuple: Quaternion components (w, x, y, z).
    """
    # Compute half-angles
    cr = np.cos(roll * 0.5)  # Cosine of roll/2
    sr = np.sin(roll * 0.5)  # Sine of roll/2
    cp = np.cos(pitch * 0.5) # Cosine of pitch/2
    sp = np.sin(pitch * 0.5) # Sine of pitch/2
    cy = np.cos(yaw * 0.5)   # Cosine of yaw/2
    sy = np.sin(yaw * 0.5)   # Sine of yaw/2
    
    # Calculate quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    return qw, qx, qy, qz

def objective(q):
    data.qpos = q
    mujoco.mj_forward(model, data)
    current_pos = data.xpos[body_id]
    current_quat = data.xquat[body_id]
    pos_error = np.linalg.norm(current_pos - desired_pos)
    quat_error = np.linalg.norm(current_quat - desired_quat)  # Simplified
    print(current_pos)
    return pos_error + quat_error

def state2transform(state, default_rotation):
    assert state.shape == (7,)
    xyz, euler, gripper_state = state[:3], state[3:6], state[6:]
    trans = RpToTrans(eulerAnglesToRotationMatrix(euler).dot(default_rotation), xyz)
    return trans

def eulerAnglesToRotationMatrix(theta):

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def RpToTrans(R, p):
    """Converts a rotation matrix and a position vector into homogeneous
    transformation matrix

    :param R: A 3x3 rotation matrix
    :param p: A 3-vector
    :return: A homogeneous transformation matrix corresponding to the inputs

    Example Input:
        R = np.array([[1, 0,  0],
                      [0, 0, -1],
                      [0, 1,  0]])
        p = np.array([1, 2, 5])
    Output:
        np.array([[1, 0,  0, 1],
                  [0, 0, -1, 2],
                  [0, 1,  0, 5],
                  [0, 0,  0, 1]])
    """
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]

def transform2quat(transform):
    print(transform)
    translation = transform[:3, 3]
    quat = quaternion_from_matrix(transform)

    return translation[0], translation[1], translation[2], quat[0], quat[1], quat[2], quat[3] # w, x, y, z

"""
python optimize_multiframe_bridge_batch_epochs_seriedata.py --model_path output/widow0
"""

NEUTRAL_JOINT_STATE = np.array([-0.13192235, -0.76238847,  0.44485444,
                                -0.01994175,  1.7564081,  -0.15953401])
DEFAULT_ROTATION = np.array([[0 , 0, 1.0],
                             [0, 1.0,  0],
                             [-1.0,  0, 0]])


if __name__ == "__main__":
    BATCH_PATH = '/data/group_data/katefgroup/datasets/bridge_chenyu/yidi/chenyu/bridge_seriedata'
    
    render_feat = False
    use_direct_feat = True
    use_direct_feat_compute = True
    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
    
    model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'model.xml'))
    chain = build_chain_from_mjcf_path(os.path.join(gaussians.model_path, 'robot_xml', 'model.xml'))
    # print(chain)
    data = mujoco.MjData(model)
    # print(data)
    n_joints = model.njnt
    site_name = 'gripper_link'  # Replace with the actual site name from your model
    body_id = model.body('wx250s/gripper_link').id
    # print(body_id, 'bodyid')
    # joint = [0.31713786,  0.04925054,  0.0539715,   0.04509302 , 0.05299974 ,-0.05953766]
    # desired_pos = np.array(joint[:3])  # Example target position [x, y, z]
    # desired_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Example target orientation (quaternion)
    # qw, qx, qy, qz = euler_to_quaternion(*joint[3:6])
    # desired_quat = np.array([qw, qx, qy, qz])

    # # Initial guess for joint angles (e.g., all zeros or a known pose)
    # initial_q = np.zeros(model.nq)

    # # Joint limits as optimization bounds
    # bounds = [(model.jnt_range[i, 0], model.jnt_range[i, 1]) for i in range(model.nq)]
    # print(bounds)

    # # Run optimization
    # result = minimize(objective, initial_q, method='SLSQP', bounds=bounds)
    # if result.success:
    #     joint_angles = result.x
    #     print("Joint angles:", joint_angles)
    # else:
    #     print("Optimization failed:", result.message)

    # time.sleep(10)
    
    if render_feat:
        my_feat_decoder = skip_feat_decoder(32).cuda()
        decoder_dict_path = os.path.join(gaussians.model_path, 'point_cloud', 'pose_conditioned_iteration_4000', 'feat_decoder.pth')
        my_feat_decoder.load_state_dict(torch.load(decoder_dict_path))
    background_color = torch.ones((3,)).cuda()
    
    SCENE_NAMEs = [d for d in os.listdir(BATCH_PATH) if os.path.isdir(os.path.join(BATCH_PATH, d))]
    SCENE_NAMEs = sorted(SCENE_NAMEs, key=lambda x: int(x[5:]))
    for i, scene_name in enumerate(SCENE_NAMEs):
        print(i, scene_name)
        assert scene_name.startswith('scene')
        # if traj_name not in ['train']:
        #     continue
        SCENE_PATH = os.path.join(BATCH_PATH, scene_name)
        image_folder = os.path.join(SCENE_PATH, 'images0')
        obs_dict = pickle.load(open(os.path.join(SCENE_PATH, 'obs_dict.pkl'), 'rb'))
        joints = obs_dict['qpos']
        print(joints.shape, 'joints shape')
        states = obs_dict["full_state"]
        print(states.shape, 'states shape')
        
        grippers_whole = states[:, -1:]
        fingers_whole = grippers_whole * 0.022 + 0.015
        fingers_whole = np.concatenate([fingers_whole, -fingers_whole], axis=1)
        joints_whole = np.concatenate([joints, fingers_whole], axis=1)
        print('joints_whole: ', joints_whole.shape)
        
        image_list_whole = []
        image_names = os.listdir(image_folder)
        image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
        
        image_paths = [os.path.join(image_folder, path) for path in image_names]
        # print(image_paths, 'image paths')
        for k in range(len(joints_whole)):
            image_path = image_paths[k]
            img = Image.open(image_path)
            img = np.array(img)
            image_list_whole.append(img)
        image_list_whole = np.stack(image_list_whole)
        B, H, W, C = image_list_whole.shape
        print(image_list_whole.shape, 'images')
        
        results_file = os.path.join(SCENE_PATH, 'results.pkl')
        extrinsics_save_path = os.path.join(SCENE_PATH, 'extrinsics.npy')
        intrinsics_save_path = os.path.join(SCENE_PATH, 'intrinsics.npy')
        vis_features = True
        tracking_loss = True
        use_sam_mask = False
        have_depth = False
        reg_mask = True
        FRANKA = True
        UR5 = False
        ROBOTIQ = True        
            
        initial_downsample = []
        l_max = min(200, len(image_list_whole))
        for i in range(l_max):
            initial_downsample.append(i)
        
        image_list_whole = image_list_whole[initial_downsample]
        joints_whole = joints_whole[initial_downsample]
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        data_file = os.path.join(SCENE_PATH, f'tracker_data.pth')
        if not os.path.exists(data_file):
            prepare_flow(image_list_whole, device, data_file)
        data_loaded = torch.load(data_file)
        fwd_flows_whole = data_loaded['fwd_flows']
        fwd_valids_whole = data_loaded['fwd_valids']
        
        # intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
        intrinsics = np.array([[557.5204, 0.0, W / 2],
                            [0.0, 557.5204, H / 2],
                            [0.0, 0.0, 1.0]])
        
        w2c = np.array([
                                [0.40230105648431419,
                                -0.54737938074000059,
                                0.73384581043452057,
                                0.0],
                                [-0.91514565098874134,
                                -0.21790723695396816,
                                0.33915317123606575,
                                0.0],
                                [-0.025735139945173187,
                                -0.8080174810137345,
                                -0.58859617136728104,
                                0.0],
                                [-0.1495516053826326,
                                0.24801468373558327,
                                0.13542126749117278,
                                1.0]
                            ]).T
        # c2w = np.linalg.inv(w2c)
        # Define perturbation ranges
        thetas_x = torch.linspace(-0.2, 0.2, 5)  # 3 steps, ±0.1 radians
        thetas_y = torch.linspace(-0.2, 0.2, 5)
        thetas_z = torch.linspace(-0.2, 0.2, 5)
        dxs = torch.linspace(-0.00, 0.00, 1)     # 3 steps, ±0.05 units
        dys = torch.linspace(-0.00, 0.00, 1)
        dzs = torch.linspace(-0.00, 0.00, 1)

        # Generate perturbed extrinsics
        perturbed_extrinsics = perturb_extrinsic(torch.tensor(w2c, dtype=torch.float32), thetas_x, thetas_y, thetas_z, dxs, dys, dzs)
        best_pose = None
        best_camera = None
        best_extrinsics = None
        best_score = float('inf')
        epoch = 0
        first_camera = None  # Track the first camera to share its parameters
        print(len(perturbed_extrinsics), 'perturbed extrinsics')
        if True:
            for extrinsic in perturbed_extrinsics:
                extrinsics = extrinsic
                optimize_camera = False
                optimize_joints = False
                camera_lr = 0.0
                joints_lr = 0.0
                optimization_steps = 1
                Optimize_list = [j for j in range(3)]
                # Optimize_list = [0, 5, 10, 15, 20, 25]
                image_list = image_list_whole[Optimize_list]
                joints = joints_whole[Optimize_list]
                grippers = grippers_whole[Optimize_list]
                fingers = fingers_whole[Optimize_list]
                fwd_flows = fwd_flows_whole[Optimize_list[:-1]]
                fwd_valids = fwd_valids_whole[Optimize_list[:-1]]
                params, flow_loss_item = compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                            fwd_flows, fwd_valids)
                if flow_loss_item < best_score:
                    best_score = flow_loss_item
                    best_extrinsics = extrinsics
            print('best pose: ', best_extrinsics)
            print(best_score, 'best score')
        # pose = np.array([0, 0.57382269, 0.84167107, -2.02259688, -0.08240233,
        #                  -2.40839443])
        # pose = np.array([0.006111708272572871, 0.7799910768896609, 0.48825174062541965, -1.5580715300648718, 0.0, -2.5771203117437884]) # for left cam
        # pose = np.array([0.14113915484291922, -0.3383929566839041, 0.8833177110592437, -2.162701547806665, -0.0, -0.9353672131449788]) # for right cam
        # extrinsics = euler_to_extrinsic(pose)
        # wait a second
        time.sleep(4)
        first_camera = None
        # extrinsics = euler_to_extrinsic(best_pose)
        extrinsics = best_extrinsics
        # extrinsics = np.array([
        #                         [0.40230105648431419,
        #                         -0.54737938074000059,
        #                         0.73384581043452057,
        #                         0.0],
        #                         [-0.91514565098874134,
        #                         -0.21790723695396816,
        #                         0.33915317123606575,
        #                         0.0],
        #                         [-0.025735139945173187,
        #                         -0.8080174810137345,
        #                         -0.58859617136728104,
        #                         0.0],
        #                         [-0.1495516053826326,
        #                         0.24801468373558327,
        #                         0.13542126749117278,
        #                         1.0]
        #                     ]).T
        # extrinsics = gt_extrinsics
        print(extrinsics)
        
        

        l = len(image_list_whole)
        # l = 100
        batch_size = 3
        step_size = 3
        steps_per_epoch = l // step_size
        epochs = 20
        offset = 0
        inset = 0
         
        for epoch in tqdm(range(epochs)):
            for batch_idx in range(steps_per_epoch):
                Optimize_list = []
                for image_idx in range(batch_size):
                    if image_idx + batch_idx * step_size + offset >= l - inset:
                        break
                    Optimize_list.append(batch_idx * step_size + image_idx + offset)
                print(Optimize_list)
                if len(Optimize_list) < 1:
                    continue
                image_list = image_list_whole[Optimize_list]
                fingers = fingers_whole[Optimize_list]
                joints = joints_whole[Optimize_list]
                fwd_flows = fwd_flows_whole[Optimize_list[:-1]]
                fwd_valids = fwd_valids_whole[Optimize_list[:-1]]
                print(image_list.shape, 'images')
        
                optimize_camera = True
                optimize_joints = False
                camera_lr = 0.001
                # decay lr using epoch
                camera_lr = camera_lr * (0.5 ** (epoch // 4))
                
                joints_lr = 0.001
                optimization_steps = 10
                powerful_optimize_dropdown = "Disabled"
                noise_input = 0.01
                num_inits_input = 1
                params, valid_optimization = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                            fwd_flows, fwd_valids)
                torch.cuda.empty_cache()
            print(params)
            results_dict = {'extrinsics': params.tolist(), 'valid_optimization': valid_optimization, 'intrinsics': intrinsics.tolist()}
            # save extrinsics and intrinsics
            print(params, 'params')
            print(intrinsics, 'intrinsics')
            np.save(extrinsics_save_path, params)
            np.save(intrinsics_save_path, intrinsics)
        