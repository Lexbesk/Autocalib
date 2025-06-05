import sys
import tempfile

"""
python optimize_multiframe_droid_batch_epochs_yidi_single.py --model_path output/franka_fr3_2f85_highres_finetune_0 --scene_path None
"""


import os
os.environ['MUJOCO_GL'] = 'egl'
import cv2
from torch.optim.lr_scheduler import StepLR

# Create own tmp directory (don't have permission to write to tmp on my cluster)
tmp_dir = os.path.join(os.getcwd(), 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir
print(f"Created temporary directory: {tmp_dir}")
os.environ['TMPDIR'] = tmp_dir

import gradio as gr

if 'notebooks' not in os.listdir(os.getcwd()):
    os.chdir('../')

import numpy as np
import mujoco
from utils_loc.mujoco_utils import simulate_mujoco_scene, compute_camera_extrinsic_matrix, extract_camera_parameters
from tqdm import tqdm
from utils_loc.generate_grid_campose import generate_camera_poses, perturb_extrinsic

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
# from optimize_multiframe_droid_batch import flow_to_color, iou_loss
def flow_to_color(flow):
    dx = flow[:, :, 0]
    dy = flow[:, :, 1]
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0, 1]
    mag = np.sqrt(dx**2 + dy**2)
    mag_max = np.max(mag)
    mag_norm = mag / mag_max if mag_max > 0 else mag
    hsv = np.stack([angle, np.ones_like(angle), mag_norm], axis=2)
    rgb = mcolors.hsv_to_rgb(hsv)
    return np.clip(rgb, 0, 1)  # Clip to [0, 1]

def iou_loss(pred, target, smooth=1e-6):
    batch_size = pred.shape[0]
    pred = pred.view(batch_size, -1)  # (batch_size, H*W)
    target = target.view(batch_size, -1)
    intersection = (pred * target).sum(dim=1)  # Sum over pixels per batch
    union = pred.sum(dim=1) + target.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return (1 - iou).mean()  # Average loss over batch

gaussians, background_color, sample_cameras, kinematic_chain = initialize_gaussians()

use_feature_loss = True

background_color = torch.zeros((3,)).cuda()
# make it orange
# background_color[0] = 1.0
# background_color[1] = 0.5
example_camera = sample_cameras[0]

model = mujoco.MjModel.from_xml_path(os.path.join(gaussians.model_path, 'robot_xml', 'model.xml'))
data = mujoco.MjData(model)
n_joints = model.njnt

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
    
    os.makedirs('inspection_features', exist_ok=True)
    if use_feature_loss:
        print("Loading DINOv2 model...")
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
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
        gaussian_feats = []
        gaussian_flows = []
        i = 0
        for camera, joint_pose in zip(all_cameras, all_joint_poses):
            if FRANKA:
                output = render_gradio(camera, gaussians, background_color, render_features = False)
                if i > 0:
                    flow_output = render_flow([all_cameras[i], all_cameras[i-1]], gaussians, background_color, render_features = False)
                    # print(flow_output["flow"].shape, 'flow output shape')
                    _, H_flow, W_flow, _ = flow_output["flow"].shape
                    flow = flow_output["flow"].squeeze().reshape((-1, 2)) # [3, H, W] -> [H*W, 3]
                    flow_2d = flow[:, 0:2]
                    flow_2d = flow_2d.reshape((H_flow, W_flow, 2)).permute(2,0,1)
                    # flow_2d = flow_2d.unsqueeze(0) # [1, 2, H, W]
                    gaussian_flows.append(flow_2d)
                    
                    # gaussian_2d_pos_curr = flow_output['gaussian_2d_pos_curr']
            elif UR5:
                output = render_gradio(camera, gaussians, background_color, render_features = False)
            
            new_image = output['render']
            new_depth = output['depth']
            # new_pcd = output['pcd']
            
            # if step == 0:
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
        if use_sam_mask:
            mujoco_mask_target = torch.stack(mujoco_masks)
        mujoco_images_target = torch.stack(mujoco_images)
        mujoco_depths_target = torch.stack(mujoco_depths)
        # print(gaussian_masks.shape, 'gaussian_masks')
        # print(gaussian_tensors.shape, 'gaussian_tensors')
        # print(gaussian_feats.shape, 'gaussian feats')
        # print(gaussian_flows.shape, 'gaussian flows')
        print(step, 'step')
        
        image = gaussian_tensors
        mujoco_images_target1 = mujoco_images_target / 255
        normalize = T.Normalize(mean=[0.5], std=[0.5])
        
        if use_feature_loss:
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
            
        if tracking_loss and step == 0:

            # print(mujoco_images_target.shape, 'mujoco images target shape')
            
            fwd_flows = fwd_flows.to(device)
            fwd_valids = fwd_valids.to(device)
                
            # print(fwd_valids)
            fwd_valids = (fwd_valids > 0.3).float()
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.5)
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
            # print(fwd_valids)
            # print(fwd_flows.shape, 'forward flow shape')
            # motion_mask = (torch.norm(fwd_flows, dim=1, keepdim=True) > 1.1)
            # print(motion_mask.sum(dim=(1, 2, 3)), 'motion mask sum')
            # motion_idx = motion_mask.sum(dim=(1, 2, 3)) > 10000
            # motion_idx = motion_idx.float().unsqueeze(1).unsqueeze(1).unsqueeze(1)
            print(motion_idx, motion_idx.shape)
            motion_mask = motion_mask.float() # [T-1, 1, H, W]
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
            
            
            
        if tracking_loss and step % 19 == 1 and epoch == 0:
            for i in range(1, min(20, mujoco_images_target.shape[0])):
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
                depth_mask_i = depth_mask[i-1, 0, :, :].float().cpu().numpy()
                depth_near_mask_i = depth_near_mask[i-1, 0, :, :].float().cpu().numpy()
                flow_valid_mask_i = fwd_valids[i-1, 0, :, :].float().cpu().numpy()
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
                axes[2, 0].imshow(motion_depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
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
                axes[3, 0].imshow(depth_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 0].set_title('Depth Mask')
                axes[3, 0].axis('off')

                axes[3, 1].imshow(new_image_np)
                axes[3, 1].imshow(depth_near_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 1].set_title('Valid Mask')
                axes[3, 1].axis('off')

                axes[3, 2].imshow(new_image_np)
                axes[3, 2].imshow(flow_valid_mask_i, cmap='gray', alpha=1, interpolation='none')
                axes[3, 2].set_title('Flow valid mask')
                axes[3, 2].axis('off')

                # Step 5: Save the figure as an image
                plt.tight_layout()  # Adjusts spacing to minimize white space
                plt.savefig(f'inspection_features/super_flow_visualization{i}_{0}.png', dpi=300)  # Save as PNG with 300 DPI
                plt.close()
    
        if tracking_loss:
            # target_mask = (self.input_mask_torch_orig_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float() if finetune_all else (self.input_mask_torch_list[b_idx-1 if b_idx > 0 else 0] > 0.5).float()

            flow_loss = (gaussian_flows - fwd_flows).abs()
            
            # We will normalize the flow loss w.r.t. to image dimensions
            flow_loss[:, 0] /= W
            flow_loss[:, 1] /= H
            # flow_loss_motion = (flow_loss * motion_mask * fwd_valids).sum() / (motion_mask * fwd_valids).sum()
            # flow_loss = (flow_loss * fwd_valids).sum() / fwd_valids.sum()
            flow_loss = (flow_loss * fwd_valids * motion_idx).sum() / fwd_valids.sum()
        
            
            # # Note that this is different from the normal balanced mask loss
            # if target_mask.sum() > 0:
            #     if (target_mask * obj_flow_valid_mask).sum() > 0:
            #         masked_flow_loss = (flow_loss * obj_flow_valid_mask * target_mask).sum() / (target_mask * obj_flow_valid_mask).sum()
            #     else:
            #         masked_flow_loss = 0.
            #     if (1 - target_mask).sum() > 0:
            #         masked_flow_loss_empty = (flow_loss * (1 - target_mask)).sum() / (1 - target_mask).sum()
            #     else:
            #         masked_flow_loss_empty = 0.
            #     loss = loss + 10000. * step_ratio * (masked_flow_loss + masked_flow_loss_empty) / self.opt.batch_size
        
        # if vis_features:
        #     for i in range(len(Optimize_list)):
        #         image = mujoco_images_target[i].permute(1, 2, 0)
        #         # print(image.shape, 'image_shape')
        #         image_np = image.to(torch.uint8)
        #         image_np = image_np.detach().cpu().numpy()
        #         image_np = Image.fromarray(image_np)
        #         image_np.save(f'inspection_features/images_{i}.png')
                
        #         image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
        #         image_np = image.to(torch.uint8)
        #         image_np = image_np.detach().cpu().numpy()
        #         image_np = Image.fromarray(image_np)
        #         image_np.save(f'inspection_features/image_render_{i}.png')
        if vis_features and step % 5 == 0:
            for i in range(len(Optimize_list)):
                image = mujoco_images_target[i].permute(1, 2, 0)
                # print(image.shape, 'image_shape')
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/images_{i}.png')
                
                image_render = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np_render = image_render.to(torch.uint8)
                image_np_render = image_np_render.detach().cpu().numpy()
                image_np_render = Image.fromarray(image_np_render)
                image_np_final = Image.fromarray(np.concatenate([image_np_render, image_np], axis=1))
                image_np_final.save(f'inspection_features/image_render_{i}.png')
            
            
        
        
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
            
        IoU_loss_motion = iou_loss(gaussian_masks[:-1], motion_depth_near_mask)
        depth_loss = F.mse_loss(mujoco_depths_target, gaussian_depths, reduction='none')
        print(depth_loss.shape, 'depth_loss shape', depth_near_mask1.shape)
        depth_diff = (depth_loss * depth_near_mask1).mean()

        # total_loss = l2_diff.sum()
        # total_loss = loss_feat + IoU_loss
        # total_loss = l2_diff.sum() + loss_feat + depth_diff * 1 + IoU_loss * 10
        flow_loss = flow_loss * 1000
        # flow_loss_motion = flow_loss_motion * 10
        depth_diff = depth_diff * 0.1
        IoU_loss_motion = IoU_loss_motion * 0.01
        # total_loss = flow_loss + flow_loss_motion
        # total_loss = depth_diff * 0.01 + flow_loss
        # if flow_loss == 0:
        #     print('Flow loss is zero, skipping optimization step.')
        #     continue
        total_loss = flow_loss + depth_diff * 0.01 + IoU_loss_motion * 0.01
        # if use_feature_loss:
        #     loss_direct_feat = loss_direct_feat * 1
        #     total_loss += loss_direct_feat
        # print('flow_loss :', flow_loss.item())
        # print('depth_diff:', depth_diff.item(),  'IoU_loss_motion :', IoU_loss_motion.item())
        # print('total_loss: ', total_loss.item())
        # print(DATA_PATH, 'epoch:', epoch)
        
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
    #     valid_masks = True
    # yield (best_image, *rounded_final_params)
    # print(all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy())
    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), valid_masks
    

def compute_loss(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, fwd_flows, fwd_valids):
    all_cameras = []
    all_joint_poses = []

    mujoco_masks = []
    mujoco_images = []
    mujoco_depths = []
    
    global first_camera
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if False:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') # s, b, l, g
        dinov2 = dinov2.to(device)
        dino_transform = T.Compose([
                                        T.ToTensor(),
                                        T.Normalize(mean=[0.5], std=[0.5]),
                                    ])
    os.makedirs('inspection_features', exist_ok=True)
    for c_idx in range(image_list.shape[0]):
        mujoco_image = image_list[c_idx]
        mujoco_tensor = torch.from_numpy(mujoco_image).float().cuda().permute(2, 0, 1).to(device, non_blocking=True)
        mujoco_images.append(mujoco_tensor)
        if use_sam_mask:
            mujoco_mask = mask_list[c_idx]
            mujoco_mask_binary = (mujoco_mask > 0).astype(np.uint8)
            mujoco_mask_tensor = torch.from_numpy(mujoco_mask_binary).cuda()
            mujoco_masks.append(mujoco_mask_tensor)
        mujoco_depth = depth_list_whole[c_idx]
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
        # print(step, 'step')
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
        
        if False:
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
                image_np.save(f'inspection_features/images_{i}.png')
                
                image = (gaussian_tensors[i] * 255).permute(1, 2, 0)
                image_np = image.to(torch.uint8)
                image_np = image_np.detach().cpu().numpy()
                image_np = Image.fromarray(image_np)
                image_np.save(f'inspection_features/image_render_{i}.png')
        threshold = 0.5
        gaussian_masks_scaled = (gaussian_masks * 255).byte()  # uint8 format
        if use_sam_mask:
            ground_truth_masks_binary = (mujoco_mask_target > threshold).float()
            ground_truth_masks_scaled = (ground_truth_masks_binary * 255).byte()
            ground_truth_masks_np = ground_truth_masks_scaled.detach().cpu().numpy()

        # Step 3: Convert to NumPy
        gaussian_masks_np = gaussian_masks_scaled.detach().cpu().numpy()
        
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
        print('flow_loss :', flow_loss.item(), 'rgb loss:', l2_diff.item())
        

    return all_cameras[0].world_view_transform.transpose(0, 1).detach().cpu().numpy(), flow_loss.item()


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


if __name__ == "__main__":
    BATCH_PATH = '/data/user_data/wenhsuac/chenyuzhang/data/droid_yidi_extract'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene_path', type=str, required=True, help='Path to the scene directory')
    parser.add_argument('--model_path', type=str, default='output/franka_fr3_2f85_highres_finetune_0', help='Path to the scene directory')
    args = parser.parse_args()
    # TRAJ_PATHs = [d for d in os.listdir(BATCH_PATH) if os.path.isdir(os.path.join(BATCH_PATH, d))]
    # TRAJ_PATHs = sorted(TRAJ_PATHs)
    TRAJ_PATH = args.scene_path
    results_file = os.path.join(TRAJ_PATH, 'results.pkl')
    results_dict = {}
    DATA_PATH = os.path.join(TRAJ_PATH)
    extrinsics_save_path = os.path.join(DATA_PATH, 'extrinsics.npy')
    intrinsics_save_path = os.path.join(DATA_PATH, 'intrinsics.npy')
    FRANKA = True
    UR5 = False
    ROBOTIQ = True
        
    cam_id = 0
    vis_features = True
    tracking_loss = True
    use_sam_mask = False
    image_list_whole = []
    depth_list_whole = []
    mask_list_whole = []
    grippers_whole = np.load(os.path.join(DATA_PATH, 'grippers.npy'))
    joints_whole = np.load(os.path.join(DATA_PATH, 'joints.npy'))
    print(grippers_whole.shape, 'grippers', grippers_whole.max(), grippers_whole.min())
    grippers_whole = np.expand_dims(grippers_whole, axis=1)
    fingers_whole = grippers_whole * np.array([[0.75, -0.4, 0.6, -0.26, 0.75, -0.4, 0.6, -0.26]])
    print(fingers_whole.shape, 'grippers')
    # fingers_whole = np.stack([grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole, grippers_whole], axis=1) # for ur5

    joints_whole = np.concatenate([joints_whole, fingers_whole], axis=1)
    print('joints_whole: ', joints_whole.shape)
    # print(joints)
    image_names = os.listdir(os.path.join(DATA_PATH, 'images0'))
    image_names = sorted(image_names, key=lambda x: int(x.split('.')[0]))
    print(image_names)
    for image_name in image_names:
        # print('image_name', image_name)
        image_path = os.path.join(DATA_PATH, 'images0', image_name)
        img = Image.open(image_path)
        img = np.array(img)
        image_list_whole.append(img)
    image_list_whole = np.stack(image_list_whole)
    _, H, W, C = image_list_whole.shape
    # print(image_list.shape, 'images')
        
    depth_names = os.listdir(os.path.join(DATA_PATH, 'depth'))
    depth_names = sorted(depth_names, key=lambda x: int(x.split('.')[0]))
    for depth_name in depth_names:
        # print('depth_name', depth_name)
        depth_path = os.path.join(DATA_PATH, 'depth', depth_name)
        depth = np.load(depth_path)
        depth_list_whole.append(depth)
    depth_list_whole = np.stack(depth_list_whole)
    
    if use_sam_mask:
        mask_names = os.listdir(os.path.join(DATA_PATH, 'masks'))
        mask_names = sorted(mask_names, key=lambda x: int(x.split('.')[0]))
        for mask_name in mask_names:
            # print('mask_name', mask_name)
            mask_path = os.path.join(DATA_PATH, 'masks', mask_name)
            mask = Image.open(mask_path)
            mask = np.array(mask)
            mask_list_whole.append(mask)
        mask_list_whole = np.stack(mask_list_whole)
    
    initial_downsample = []
    for i in range(len(image_list_whole) // 2, len(image_list_whole)):
        initial_downsample.append(i)
    
    image_list_whole = image_list_whole[initial_downsample]
    depth_list_whole = depth_list_whole[initial_downsample]
    if use_sam_mask:
        mask_list_whole = mask_list_whole[initial_downsample]
    joints_whole = joints_whole[initial_downsample]
    grippers_whole = grippers_whole[initial_downsample]
    fingers_whole = fingers_whole[initial_downsample]
    
    
    if use_sam_mask:
        print(mask_list_whole.shape, 'masks ensure not 3 channels')
    print(grippers_whole.shape, 'grippers')
    print(joints_whole.shape, 'joints')
    print(image_list_whole.shape, 'images')
    print(depth_list_whole.shape, 'depths')
    assert(depth_list_whole.shape[0] == image_list_whole.shape[0] == grippers_whole.shape[0] == joints_whole.shape[0])
    print('frames: ', image_list_whole.shape[0])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    data_file = os.path.join(DATA_PATH, 'tracker_data.pth')
    if not os.path.exists(data_file):
        prepare_flow(image_list_whole, device, data_file)
    data_loaded = torch.load(data_file)
    fwd_flows_whole = data_loaded['fwd_flows']
    fwd_valids_whole = data_loaded['fwd_valids']
    
    Current_Cams = {'19824535_left': np.array([-0.07528311, 0.02872137, 0.02331657, -0.32557736, 0.00624743, -1.56287453]),
                    '19824535_right': np.array([-0.06961441, -0.0353966, 0.02041079, -0.31886274, 0.00782202, -1.55214306]),
                    '22008760_left': np.array([0.43518401, 0.37382269, 0.34167107, -2.02259688, -0.08240233, -2.40839443]),
                    '22008760_right': np.array([0.34059767, 0.29106813, 0.35430316, -2.02269023, -0.07155337, -2.40794443]),
                    '23404442_left': np.array([0.25955495, -0.50010183, 0.4368819, -2.27734563, 0.33618977, -0.5369007]),
                    '23404442_right': np.array([0.35203881, -0.55772972, 0.40182455, -2.27980636, 0.34639743, -0.54752158]),
                    '24400334_left': np.array([2.59675732e-01, -3.66262596e-01, 2.48493048e-01, -1.74211540e+00, -1.21274269e-03, -7.14986722e-01]),
                    '24400334_right': np.array([0.35026008, -0.44562786, 0.25002654, -1.74644264, -0.00601653, -0.71559287]),
                    '26405488_left': np.array([0.11133259, 0.82074817, 0.04550905, -1.64633975, -0.17857016, -2.93766368]),
                    '26405488_right': np.array([-0.00941784, 0.79726688, 0.06931512, -1.64730253, -0.18917295, -2.9422189]),
                    '29838012_left': np.array([0.4154884, 0.4775471, 0.41143223, -2.21604102, -0.02257695, -2.96372386]),
                    '29838012_right': np.array([0.29402237, 0.45672118, 0.41110094, -2.20799482, -0.0194012, -2.95977621])}
    
    Optimal_param = {
        '22008760_left': {'w': torch.tensor([-0.0,  0.0, -0.0]), 'v': torch.tensor([ 0., -0., -0.])}, 
    }
    
    # assert cam_id in Current_Cams.keys(), f'cam_id {cam_id} not in Current_Cams'
    intrinsics = np.load(os.path.join(DATA_PATH, 'intrinsics.npy'))
    extrinsics = np.load(os.path.join(DATA_PATH, 'extrinsics0.npy'))
    
    thetas_x = torch.linspace(-0.2, 0.2, 5)  # 3 steps, ±0.1 radians
    thetas_y = torch.linspace(-0.2, 0.2, 5)
    thetas_z = torch.linspace(-0.1, 0.1, 4)
    dxs = torch.linspace(-0.00, 0.00, 1)     # 3 steps, ±0.05 units
    dys = torch.linspace(-0.00, 0.00, 1)
    dzs = torch.linspace(-0.03, 0.03, 3)

    # Generate perturbed extrinsics
    perturbed_extrinsics = perturb_extrinsic(torch.tensor(extrinsics, dtype=torch.float32), thetas_x, thetas_y, thetas_z, dxs, dys, dzs)
    best_pose = None
    best_camera = None
    best_extrinsics = None
    best_score = float('inf')
    epoch = 0
    first_camera = None  # Track the first camera to share its parameters
    print(len(perturbed_extrinsics), 'perturbed extrinsics')
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

    l = len(image_list_whole)
    print(l, 'length of images')
    batch_size = 3
    steps_per_epoch = l // batch_size
    offset = 10

    epochs = 20
    
    for epoch in tqdm(range(epochs)):
        for batch_idx in range(steps_per_epoch):
            Optimize_list = []
            for image_idx in range(batch_size):
                if image_idx + batch_idx * batch_size + offset >= l:
                    break
                Optimize_list.append(batch_idx * batch_size + image_idx + offset)
            print(Optimize_list)
            if len(Optimize_list) < batch_size - 1:
                continue
            image_list = image_list_whole[Optimize_list]
            depth_list = depth_list_whole[Optimize_list]
            if use_sam_mask:
                mask_list = mask_list_whole[Optimize_list]
            joints = joints_whole[Optimize_list]
            grippers = grippers_whole[Optimize_list]
            fingers = fingers_whole[Optimize_list]
            fwd_flows = fwd_flows_whole[Optimize_list[:-1]]
            fwd_valids = fwd_valids_whole[Optimize_list[:-1]]
            # print(image_list.shape, 'images')
            # print(depth_list.shape, 'depths')
            # print(mask_list.shape, 'masks')
            # print(grippers.shape, 'grippers')
            # print(joints.shape, 'joints')
            # print(fingers.shape, 'fingers')
    
            optimize_camera = True
            optimize_joints = False
            camera_lr = 0.002
            # decay lr using epoch
            camera_lr = camera_lr * (0.5 ** (epoch // 4))
            
            joints_lr = 0.001
            optimization_steps = 10
            powerful_optimize_dropdown = "Disabled"
            noise_input = 0.01
            num_inits_input = 1
            params, valid_optimization = optimize(optimize_camera, optimize_joints, camera_lr, joints_lr, optimization_steps, 
                        fwd_flows, fwd_valids)
        # print(params)
        results_dict['0'] = {'extrinsics': params.tolist(), 'valid_optimization': valid_optimization}
        with open(results_file, 'wb') as f:
            pickle.dump(results_dict, f) 
        # print(params, 'params')
        # print(intrinsics, 'intrinsics')
        np.save(extrinsics_save_path, params)
        # np.save(intrinsics_save_path, intrinsics)
    