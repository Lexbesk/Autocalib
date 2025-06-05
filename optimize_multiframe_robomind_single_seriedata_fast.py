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
# if 'notebooks' not in os.listdir(os.getcwd()):
#     os.chdir('../')
    
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import gc

import numpy as np
from utils_loc.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from utils_loc.generate_grid_campose import generate_camera_poses, perturb_extrinsic
from tqdm import tqdm
import argparse

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
from utils_loc.generate_grid_campose import generate_camera_poses


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
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_direct_feat:
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
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        
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
            if cam_id in Optimal_param:
                Optimal_Camera_Param = Optimal_param[cam_id]
                first_camera.w.data = Optimal_Camera_Param['w'].to(camera.device)
                first_camera.v.data = Optimal_Camera_Param['v'].to(camera.device)
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
        # gaussian_depths = torch.where(gaussian_depths == 0, 4.0, gaussian_depths)
        if tracking_loss:
            gaussian_flows = torch.stack(gaussian_flows)
        if tracking_loss and step == 0:
            mujoco_images_target = torch.stack(mujoco_images)
            
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
            if motion_idx.sum() <= 0:
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!valid images < 0 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                have_motion = False
                torch.cuda.empty_cache()
                break
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
            
        # Visualization
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
            flow_loss = (gaussian_flows - fwd_flows).abs()
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
            # if not have_motion:
            #     flow_loss *= 0
        
        if vis_features and step % 5 == 0:
            for i in range(2):
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
                print(f'inspection_features_1/image_render_{i}.png saved')
        # if reg_mask:
        #     gaussian_mask_sum = gaussian_masks.sum()
        #     gaussian_mask_sum = gaussian_mask_sum / gaussian_masks.shape[0]
        #     reg_mask_loss = (10000 - gaussian_mask_sum)
        #     print(reg_mask_loss.item(), 'reg mask loss')
        
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()

        flow_loss = flow_loss * 1000
        total_loss = flow_loss + l2_diff * 0.1
        # total_loss = 0
        if use_direct_feat:
            loss_direct_feat = loss_direct_feat * 1
            total_loss += loss_direct_feat
        # print('flow_loss :', flow_loss.item())
        # if use_direct_feat:
        #     print('loss_feat:', loss_direct_feat.item())
        # print('rgb loss:', l2_diff.item())
        # print('total_loss: ', total_loss.item())
        print(TRAJ_PATH, 'epoch:', epoch)
        
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
    

# for grid search only
def compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if use_direct_feat_compute:
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
            if cam_id in Optimal_param:
                Optimal_Camera_Param = Optimal_param[cam_id]
                first_camera.w.data = Optimal_Camera_Param['w'].to(camera.device)
                first_camera.v.data = Optimal_Camera_Param['v'].to(camera.device)
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
            # new_pcd = output['pcd']
            
            # if step == 0 and i <= 0:
            #     # print(f'-----------------------------------saving pcd from {i}')
            #     os.makedirs("pcd", exist_ok=True)
            #     np.save(f"pcd/means3D_{i}.npy", new_pcd)
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
            mujoco_images_target = torch.stack(mujoco_images)
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
            # print(mujoco_depths_target)
            # print(depth_mask)
            depth_mask = depth_mask.unsqueeze(1).float() # [T-1, 1, H, W]
            
            # print(depth_mask.shape, 'depth mask shape')
            # print(motion_mask)
            # print(motion_mask.shape, 'motion mask shape')
            
            
            
        if tracking_loss and step % 19 == 1 and epoch == 0:
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
                plt.savefig(f'inspection_features_1/super_flow_visualization{i}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
                
        # image = gaussian_tensors
        # mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])

        if use_direct_feat_compute:
            print(gaussian_tensors.shape, 'gaussian tensors shape')
            resized_image = resize_tensor(gaussian_tensors, longest_edge=800)
            resized_mujo_image = resize_tensor(mujoco_images_target / 255, longest_edge=800)
            
            image_final = normalize(resized_image)
            mujo_final = normalize(resized_mujo_image)
            print(dinov2.patch_size, 'dino patch size')
            dino_image, target_H, target_W = interpolate_to_patch_size(image_final, dinov2.patch_size)
            dino_mojo, _, _ = interpolate_to_patch_size(mujo_final, dinov2.patch_size)
            features = dinov2.forward_features(dino_image)["x_norm_patchtokens"]
            mujo_features = dinov2.forward_features(dino_mojo)["x_norm_patchtokens"]
            p = dinov2.patch_size  # Typically 14 for DINOv2
            grid_h = target_H // p  # Number of patches along height
            grid_w = target_W // p  # Number of patches along width
            
            # Reshape to (batch_size, feature_dim, grid_h, grid_w) for pooling
            feature_dim = features.shape[-1]  # e.g., 768
            features_reshaped = features.view(1, grid_h, grid_w, feature_dim).permute(0, 3, 1, 2)
            mujo_features_reshaped = mujo_features.view(1, grid_h, grid_w, feature_dim).permute(0, 3, 1, 2)
            
            # Apply 2x2 average pooling to reduce spatial dimensions by half
            pool = torch.nn.AvgPool2d(kernel_size=4, stride=4)
            features_pooled = pool(features_reshaped)
            mujo_features_pooled = pool(mujo_features_reshaped)
            
            # Reshape back to (batch_size, new_num_patches, feature_dim)
            new_grid_h = features_pooled.shape[2]  # Reduced height
            new_grid_w = features_pooled.shape[3]  # Reduced width
            features_pooled = features_pooled.permute(0, 2, 3, 1).view(1, new_grid_h * new_grid_w, feature_dim)
            mujo_features_pooled = mujo_features_pooled.permute(0, 2, 3, 1).view(1, new_grid_h * new_grid_w, feature_dim)
            
            # Compute the loss with downsampled features
            loss_direct_feat = cosine_loss(features_pooled, mujo_features_pooled)
            # loss_direct_feat = cosine_loss(features, mujo_features)
            print(loss_direct_feat.item(), 'direct feature loss')
            
        
        if tracking_loss:
            flow_loss = (gaussian_flows - fwd_flows).abs()
            
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
        
        if vis_features:
            for i in range(2):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/images_{i}.png')
                
                image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/image_render_{i}.png')
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        

        # Step 4: Save masks as images
        os.makedirs('masks_inspection', exist_ok=True)
        for i in range(gaussian_masks_np.shape[0]):
            gaussian_filename = f"masks_inspection/gaussian_mask_{i}.png"
            cv2.imwrite(gaussian_filename, gaussian_masks_np[i])

        
        mse_loss = F.mse_loss(mujoco_images_target / 255, gaussian_tensors, reduction='none')
        l2_diff = mse_loss.mean(dim=(1, 2, 3)).mean()
        

        flow_loss = flow_loss * 1000 
        if flow_loss == 0:
            return None, float('inf')
        if use_direct_feat_compute:
            print('flow_loss :', flow_loss.item(), 'loss_feat:', loss_direct_feat.item(), 'rgb loss:', l2_diff.item())
        else:
            print('flow_loss :', flow_loss.item(), 'rgb loss:', l2_diff.item())
        

    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), flow_loss.item()


def prepare_flow(images, device, data_file):
    mujoco_images = []
    for c_idx in range(len(images)):
        mujoco_image = images[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
    mujoco_images_target = torch.stack(mujoco_images)
    # cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
    # Run cotracker
    print('here run tracker')
    fwd_flows, fwd_valids = run_tracker_on_images(cotracker, mujoco_images_target, n_frame=1) # T-1 2 H W, T-1 H W
    fwd_valids = fwd_valids.unsqueeze(1)
    # Prepare the data by moving tensors to CPU and storing in a dictionary
    data_to_save = {
        'fwd_flows': fwd_flows.detach().cpu(),
        'fwd_valids': fwd_valids.detach().cpu()
    }
    # Save the dictionary to the file
    torch.save(data_to_save, data_file)
        

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


"""
python optimize_multiframe_robomind_single_seriedata.py --model_path output/franka_fr3_2f85_highres_finetune_0 --scene_path /data/group_data/katefgroup/datasets/robomind/robomind_chenyu/robomind_extract_0/scene_0
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene directory')
    parser.add_argument('--model_path', type=str, default='output/franka_fr3_2f85_highres_finetune_0', help='Path to the scene directory')
    args = parser.parse_args()
    use_direct_feat = True
    use_direct_feat_compute = False
    gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()
    
    TRAJ_PATH = args.scene_path
    BATCH_PATH = os.path.dirname(TRAJ_PATH)
    print('TRAJ_PATH: ', TRAJ_PATH)
    print('BATCH_PATH: ', BATCH_PATH)
    results_file = os.path.join(BATCH_PATH, 'results.pkl')
    try:
        with open(results_file, 'rb') as f:
            results_dict = pickle.load(f)  # results_dict should have keys like "scene_0", with value being the extrinsics matrix (w2c)
    except FileNotFoundError:
        # print(f"File {results_file} not found.")
        results_dict = {}

    TRAJ_PATH = TRAJ_PATH
    extrinsics_save_path = os.path.join(TRAJ_PATH, 'extrinsics.npy')
    FRANKA = True
    ROBOTIQ = True
            
    cam_id = 0
    vis_features = True
    tracking_loss = True
    reg_mask = True
    image_list_whole = []
    depth_list_whole = []
    mask_list_whole = []
    grippers_whole = np.load(os.path.join(TRAJ_PATH, 'grippers.npy'))
    joints_whole = np.load(os.path.join(TRAJ_PATH, 'joints.npy'))
    fingers_whole = grippers_whole * np.array([[0.75, -0.4, 0.6, -0.26, 0.75, -0.4, 0.6, -0.26]])
    joints_whole = np.concatenate([joints_whole, fingers_whole], axis=1)
    image_names = os.listdir(os.path.join(TRAJ_PATH, 'images0'))
    image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
    print(image_names, 'image names')
    for image_name in image_names:
        image_path = os.path.join(TRAJ_PATH, 'images0', image_name)
        img = Image.open(image_path)
        img = np.array(img)
        image_list_whole.append(img)
    image_list_whole = np.stack(image_list_whole)
    _, H, W, C = image_list_whole.shape
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    L = len(image_list_whole)
    skip_step = L // 6
    optimize_image_ids = []
    for i in range(2, L, skip_step):
        if i >= L - 2:
            break
        optimize_image_ids.append(i)
    
    optimize_image_ids = [(idx, idx+1) for idx in optimize_image_ids]
    print(optimize_image_ids, 'optimize image ids')
    
    Optimal_param = {
        '22008760_left': {'w': torch.tensor([-0.0,  0.0, -0.0]), 'v': torch.tensor([ 0., -0., -0.])}, 
    }
    
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device, non_blocking=True)
    os.makedirs(os.path.join(TRAJ_PATH, 'tracker_data'), exist_ok=True)
    for idx, idx_next in tqdm(optimize_image_ids):
        image_pair = image_list_whole[idx:idx_next+1]    
        data_file = os.path.join(TRAJ_PATH, 'tracker_data', f'tracker_data_{idx}_{idx_next}.pth')
        # print(data_file, 'data file')
        # if not os.path.exists(data_file):
        prepare_flow(image_pair, device, data_file)

    intrinsics = np.load(os.path.join(TRAJ_PATH, 'intrinsics.npy'))
    extrinsics = np.load(os.path.join(TRAJ_PATH, 'extrinsics0.npy')) # initial guess of camera pose
    
    thetas_x = torch.linspace(-0.2, 0.2, 3)  # 3 steps, ±0.1 radians
    thetas_y = torch.linspace(-0.2, 0.2, 3)
    thetas_z = torch.linspace(-0.1, 0.1, 2)
    dxs = torch.linspace(-0.00, 0.00, 1)     # 3 steps, ±0.05 units
    dys = torch.linspace(-0.00, 0.00, 1)
    dzs = torch.linspace(-0.00, 0.00, 1)

    # Generate perturbed extrinsics
    perturbed_extrinsics = perturb_extrinsic(torch.tensor(extrinsics, dtype=torch.float32), thetas_x, thetas_y, thetas_z, dxs, dys, dzs)
    best_pose = None
    best_camera = None
    best_extrinsics = None
    best_score = float('inf')
    epoch = 0
    first_camera = None  # Track the first camera to share its parameters
    print(len(perturbed_extrinsics), 'perturbed extrinsics')
    
    # Do grid search or not
    if False:
        for extrinsic in perturbed_extrinsics:
            extrinsics = extrinsic
            optimize_camera = False
            optimize_joints = False
            camera_lr = 0.0
            joints_lr = 0.0
            optimization_steps = 1
            Optimize_list = [j for j in range(10)]
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
    
        # extrinsics = euler_to_extrinsic(best_pose)
        extrinsics = best_extrinsics
        
        


        
    first_camera = None
    epochs = 16
    
    for epoch in tqdm(range(epochs)):
        for idx, idx_next in optimize_image_ids:
            data_file = os.path.join(TRAJ_PATH, 'tracker_data', f'tracker_data_{idx}_{idx_next}.pth')
            data_loaded = torch.load(data_file)
            fwd_flows = data_loaded['fwd_flows']
            fwd_valids = data_loaded['fwd_valids']
            # print(fwd_flows_whole.shape, 'fwd flows shape')
            # print(fwd_valids_whole.shape, 'fwd valids shape')
            image_list = image_list_whole[idx:idx_next+1]
            joints = joints_whole[idx:idx_next+1]
            grippers = grippers_whole[idx:idx_next+1]
            fingers = fingers_whole[idx:idx_next+1]
    
            optimize_camera = True
            optimize_joints = False
            camera_lr = 0.004
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
        results_dict['0'] = {'extrinsics': params.tolist(), 'valid_optimization': valid_optimization}
        with open(results_file, 'wb') as f:
            print('writing results to file')
            pickle.dump(results_dict, f)
        # Save the extrinsics
        np.save(extrinsics_save_path, params)
            

del image_list_whole
del depth_list_whole
del mask_list_whole
del grippers_whole
del joints_whole
del fingers_whole
gc.collect()    